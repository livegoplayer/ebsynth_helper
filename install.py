import launch

#todo: check if ffmpeg-python is used anywhere, could have sworn i removed it but not 100% enough to change now.
if not launch.is_installed("ffmpeg-python"):
    launch.run_pip("install ffmpeg-python", "requirements for EbsyntHelper extension")

if not launch.is_installed("moviepy"):
    launch.run_pip("install moviepy", "requirements for EbsyntHelper extension")
    
if not launch.is_installed("imageio_ffmpeg"):
    launch.run_pip("install imageio_ffmpeg", "requirements for EbsyntHelper extension")

if not launch.is_installed("scenedetect"):
    launch.run_pip("install scenedetect", "requirements for EbsyntHelper extension")

if not launch.is_installed("modnet-entry"):
    launch.run_pip("install git+https://github.com/RimoChan/modnet-entry.git", "requirements for EbsyntHelper extension")

launch.git_clone("https://github.com/isl-org/MiDaS.git", "repositories/midas", "midas", "1645b7e")
