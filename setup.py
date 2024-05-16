from setuptools import setup

APP = ['gui.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'packages': [
        'os', 'random', 'time', 'tkinter', 'pandas', 'sklearn', 'matplotlib', 'minizinc'
    ],
    # Additional options can be specified if necessary
    'includes': ['numpy', 'matplotlib', 'sklearn', 'pandas'],
    'excludes': ['scipy'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)

