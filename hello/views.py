from django.http import FileResponse, JsonResponse
from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import serializers
import json
import shutil
import tempfile
import os
import zipfile


def home(request):
    return HttpResponse("Hello, Django!")


def my_view(request):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT TOP 10 * FROM accessLogs order by ID_COLUMN DESC")
        rows = cursor.fetchall()

    # Process rows or pass them to the template context
    context = {'rows': rows}
    return render(request, 'my_template.html', context)

# Define a serializer to handle the raw data.


class RawDataSerializer(serializers.Serializer):
    def to_representation(self, instance):
        # Convert row tuple to dictionary
        fields = [field[0] for field in self.context['cursor'].description]
        return dict(zip(fields, instance))


@api_view(['GET'])
def getAllLogs(request):
    try:
        with connection.cursor() as cursor:
            # Your raw SQL query
            sql = """

                 SELECT TOP 100
                       AL.[ID_column]  id_Column
                      ,AL.[PageName]   pageName
                      ,AL.[AccessDate] accessDate
                      ,AL.[IpValue]    ipValue
                 FROM 
                       [dbo].[accessLogs] AL
                    WHERE
                       AL.[LogType] = 1
                    AND
                        (AL.PAGENAME LIKE '%DEMO%'
                    and
                        AL.PAGENAME LIKE '%PAGE%')
                    AND
                        AL.PAGENAME NOT LIKE '%ERROR%'
                    AND 
                        AL.PAGENAME  NOT LIKE '%PAGE_DEMO_INDEX%'
                    AND 
                        UPPER(AL.PAGENAME) NOT LIKE '%CACHE%'
                    AND
                        AL.IPVALUE <> '::1'
                 order by 
                       AL.[ID_column] desc
                  """
            cursor.execute(sql)
            rows = cursor.fetchall()
            serializer = RawDataSerializer(rows, many=True, context={
                                           'cursor': cursor})  # Pass cursor for field names
            return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def getAllPersons(request):
    try:
        with connection.cursor() as cursor:
            # Your raw SQL query
            sql = """

                  SELECT 
                     [Id_Column]        id_Column
                    ,[NombreCompleto]   nombreCompleto
                    ,[ProfesionOficio]  profesionOficio
                    ,[Ciudad]           ciudad
                FROM
                    [dbo].[Persona]
                ORDER BY 
                    Id_Column 

                """
            cursor.execute(sql)
            rows = cursor.fetchall()
            serializer = RawDataSerializer(rows, many=True, context={
                                           'cursor': cursor})  # Pass cursor for field names
            return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def getAllContactForms(request):
    try:
        with connection.cursor() as cursor:
            # Your raw SQL query
            sql = """

                  SELECT 
                        id         id 
                        ,Name      name
                        ,Email     field_1
                        ,Message   field_2
                        ,CreatedAt field_3
                FROM
                    ContactForm
                ORDER BY 
                    id desc
                """
            cursor.execute(sql)
            rows = cursor.fetchall()
            serializer = RawDataSerializer(rows, many=True, context={
                                           'cursor': cursor})  # Pass cursor for field names
            return Response(serializer.data)
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def train_tictactoe_model(request):
    try:
        # === Force Compatibility Mode ===
        import tensorflow as tf
        import numpy as np

        # Disable V2 behavior where possible (reduces op complexity)
        # tf.compat.v1.disable_v2_behavior()  # Helps with older runtime compatibility

        # Enforce CPU-only (avoid GPU-specific kernels)
        tf.config.set_visible_devices([], 'GPU')

        print("Using TensorFlow version:", tf.__version__)

    except ImportError:
        return JsonResponse({'error': 'Failed to import TensorFlow. Make sure it is installed.'}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'TensorFlow setup failed: {str(e)}'}, status=500)

    try:
        # === STEP 1: Generate synthetic self-play data ===
        def generate_game():
            board = np.zeros(9, dtype=int)
            moves = []
            turn = 1  # X starts
            while True:
                empty = np.where(board == 0)[0]
                if len(empty) == 0:
                    break
                move = np.random.choice(empty)
                moves.append((board.copy(), move))
                board[move] = turn

                # Check win conditions
                b = board.reshape(3, 3)
                win_patterns = [
                    b[0, :], b[1, :], b[2, :],           # Rows
                    b[:, 0], b[:, 1], b[:, 2],           # Columns
                    [b[0, 0], b[1, 1], b[2, 2]],         # Diagonal \
                    [b[0, 2], b[1, 1], b[2, 0]]          # Diagonal /
                ]
                won = False
                for pattern in win_patterns:
                    if all(p == turn for p in pattern):
                        won = True
                        break
                if won:
                    break
                turn = -turn
            return [(state.flatten(), move) for state, move in moves]

        # Generate dataset
        X_train = []
        y_train = []
        for _ in range(3000):  # ~27k moves
            game = generate_game()
            for state, move in game:
                X_train.append(state)
                y_train.append(move)

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)

        # === STEP 2: Build Model (Simple Sequential) ===
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu',
                                  input_shape=(9,), name='dense_1'),
            tf.keras.layers.Dense(64, activation='relu', name='dense_2'),
            tf.keras.layers.Dense(9, activation='softmax', name='output')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # === STEP 3: Save Model in Compatible Format ===
        save_dir = "tictactoe_tf_model"
        if os.path.exists(save_dir):
            # Clean old model
            import shutil
            shutil.rmtree(save_dir)

        # Use low-level SavedModel save for maximum compatibility
        @tf.function
        def predict_fn(x):
            return model(x)

        # Define input signature explicitly
        input_spec = tf.TensorSpec(
            shape=[None, 9], dtype=tf.float32, name="input_tensor")
        concrete_function = predict_fn.get_concrete_function(input_spec)

        # Save with minimal signatures
        tf.saved_model.save(
            model,
            save_dir,
            signatures=concrete_function,
            options=None  # Avoid new serialization features
        )

        return JsonResponse({
            'status': 'success',
            'message': 'Model trained and saved using TF 2.8.0 with backward compatibility.',
            'epochs': 10,
            'dataset_size': len(X_train),
            'save_path': save_dir,
            'tensorflow_version': tf.__version__
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
def download_tictactoe_model(request):
    model_dir = "tictactoe_tf_model"
    if not os.path.exists(model_dir):
        return JsonResponse({"error": "Model not trained yet."}, status=404)

    try:
        # Create a temporary directory for the zip file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, 'tictactoe_tf_model.zip')

        # Create ZIP archive
        shutil.make_archive(
            # output path (no extension)
            base_name=os.path.join(temp_dir, 'tictactoe_tf_model'),
            format='zip',
            root_dir=model_dir  # directory to compress
        )

        # Serve the file
        response = FileResponse(
            open(zip_path, 'rb'),
            content_type='application/zip'
        )
        response['Content-Disposition'] = 'attachment; filename="tictactoe_tf_model.zip"'
        return response

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
