# speedkit/sitecustomize.py
# Wordt automatisch geïmporteerd door Python als deze dir op PYTHONPATH staat.
# Doel: GUI-popups uitzetten in child-process (cv2.imshow/plt.show), zodat batch non-interactive is.

import os

if os.environ.get("NOGUI", ""):
    # OpenCV: alle show-calls negeren
    try:
        import cv2  # type: ignore
        def _noop(*args, **kwargs): 
            return None
        cv2.imshow = _noop
        cv2.waitKey = lambda *a, **k: 1
        cv2.destroyAllWindows = _noop
    except Exception:
        pass

    # Matplotlib headless
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        def _noop(*args, **kwargs):
            return None
        plt.show = _noop
    except Exception:
        pass
