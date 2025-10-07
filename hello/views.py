import tempfile
import shutil
import os
import tensorflow as tf
import numpy as np
from rest_framework import serializers
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.db import connection
from django.http import HttpResponse
from django.shortcuts import render
from django.http import FileResponse, JsonResponse


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

# === Generate synthetic self-play data ===
def generate_game():
    board = np.zeros(9, dtype=int)
    moves = []
    turn = 1
    while True:
        empty = np.where(board == 0)[0]
        if len(empty) == 0:
            break
        move = np.random.choice(empty)
        moves.append((board.copy(), move))
        board[move] = turn

        b = board.reshape(3, 3)
        win_patterns = [
            b[0,:], b[1,:], b[2,:],
            b[:,0], b[:,1], b[:,2],
            [b[0,0], b[1,1], b[2,2]],
            [b[0,2], b[1,1], b[2,0]]
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
    for _ in range(3000):
        game = generate_game()
        for state, move in game:
            X_train.append(state)
            y_train.append(move)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Save model
    save_dir = "models/tictactoe_tf_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save(save_dir)

    return save_dir

@api_view(['GET'])
def train_tictactoe_model(request):
    try:
        # Call your function
        save_dir = generate_game()

        return JsonResponse({
            'status': 'success',
            'message': 'Model trained and saved!',
            'save_path': 'tictactoe_tf_model'
        })

    except Exception as e:
        # ✅ Also return error response
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
def _train_tictactoe_model(request):
    try:
        # Force pure-Python protobuf implementation
        import os
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

        import tensorflow as tf
        import numpy as np
    except ImportError as e:
        return JsonResponse({'error': f'Missing dep: {e.name}'}, status=500)

    # Confirm eager mode
    if not tf.executing_eagerly():
        return JsonResponse({'error': 'Eager execution required'}, status=500)

    try:
        # ... your game generation and training logic ...

        # Save model safely
        model.save("tictactoe_tf_model", save_format="tf")

        return JsonResponse({
            'status': 'success',
            'message': 'Model trained and saved!',
            'save_path': 'tictactoe_tf_model'
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
