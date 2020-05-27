import os, pathlib
import pytest

os.chdir( pathlib.Path.cwd() / 'tests/' )

pytest.main(['--cov', 'neuralpy/'])