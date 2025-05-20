from roboflow import Roboflow

rf = Roboflow(api_key="Xe7uq5tdgM8eHuFYPTa1")
project = rf.workspace("aiacworkspace").project("ai-academy-pre-final")
version = project.version(1)
dataset = version.download("yolov5")
print("Done!")