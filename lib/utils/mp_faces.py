import mediapipe as mp
from pathlib import Path

FACES_FILE_PATH = Path(__file__).resolve().parent / "mediapipe_faces.txt"


def _generate_faces():
    if FACES_FILE_PATH.exists():
        return
    edges = mp.solutions.face_mesh.FACEMESH_TESSELATION
    # connexions[i][j] true iff i and j are connected
    connexions = [[False for i in range(468)] for j in range(468)]  # 468 is the number of landmarks
    for edge in edges:
        connexions[edge[0]][edge[1]] = True
        connexions[edge[1]][edge[0]] = True
    with open(FACES_FILE_PATH, "w") as f:
        for edge in edges:
            i, j = edge
            for k in range(468):
                if connexions[i][k] and connexions[j][k]:
                    f.write(f"{i} {j} {k}\n")


def mediapipe_faces():
    _generate_faces()
    with open(FACES_FILE_PATH, "r") as f:
        faces = f.read().splitlines()
        faces = [tuple(map(int, face.split())) for face in faces]
    return faces
