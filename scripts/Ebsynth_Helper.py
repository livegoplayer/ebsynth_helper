import os
import glob
#om nom nom nom
import requests
import json
from pprint import pprint
import base64
import numpy as np
from io import BytesIO
import extensions.ebsynth_helper.scripts.berry_utility
import scripts.optical_flow_simple as opflow
from PIL import Image, ImageOps,ImageFilter
import io
from collections import deque
import cv2
import scripts.Berry_Method as bmethod
import scripts.berry_utility as butility
import re



def sort_into_folders(video_path, fps, per_side, batch_size, _smol_resolution,square_textures,max_frames,output_folder,border):
    border = 0
    per_batch_limmit = (((per_side * per_side) * batch_size)) + border

    frames = []
  #  original_frames_directory = os.path.join(output_folder, "original_frames")
  #  if os.path.exists(original_frames_directory):
  #      for filename in os.listdir(original_frames_directory):
  #          frames.append(cv2.imread(os.path.join(original_frames_directory, filename), cv2.COLOR_BGR2RGB))
  #  else:
    video_data = bmethod.convert_video_to_bytes(video_path)
    frames = butility.extract_frames_movpie(video_data, fps, max_frames)
    
    print(f"full frames num = {len(frames)}")


    output_frames_folder = os.path.join(output_folder, "frames")
    if not os.path.exists(output_frames_folder):
        os.makedirs(output_frames_folder)
    output_keys_folder = os.path.join(output_folder, "keys")
    if not os.path.exists(output_keys_folder):
        os.makedirs(output_keys_folder)
    input_folder = os.path.join(output_folder, "input")

    filenames = os.listdir(input_folder)
    img = Image.open(os.path.join(input_folder, filenames[0]))
    original_width, original_height = img.size
    height,width = frames[0].shape[:2]
    
    texture_aspect_ratio = float(width) / float(height)
    
    
    _smol_frame_height = _smol_resolution
    _smol_frame_width = int(_smol_frame_height * texture_aspect_ratio)
    print(f"saving size = {_smol_frame_width}x{_smol_frame_height}")


    for i, frame in enumerate(frames):
        frame_to_save = cv2.resize(frame, (_smol_frame_width, _smol_frame_height), interpolation=cv2.INTER_LINEAR)
        bmethod.save_square_texture(frame_to_save, os.path.join(output_frames_folder, "frames{:05d}.png".format(i)))
    original_frame_height,original_frame_width = frames[0].shape[:2]
    


    bigbatches,frameLocs = bmethod.split_frames_into_big_batches(frames, per_batch_limmit,border,ebsynth=True,returnframe_locations=True)
    bigprocessedbatches = []

    last_frame_end = 0
    print (len(square_textures))
    for a,bigbatch in enumerate(bigbatches):
        batches = bmethod.split_into_batches(bigbatches[a], batch_size,per_side* per_side)

        keyframes = [batch[int(len(batch)/2)] for batch in batches]
        if a < len(square_textures):
            resized_square_texture = cv2.resize(square_textures[a], (original_width, original_height), interpolation=cv2.INTER_LINEAR)
            new_frames = bmethod.split_square_texture(resized_square_texture,len(keyframes), per_side* per_side,_smol_resolution,True)
            new_frame_start,new_frame_end = frameLocs[a]
            
            for b in range(len(new_frames)):
                print (new_frame_start)
                inner_start = last_frame_end
                inner_end = inner_start + len(batches[b])
                last_frame_end = inner_end
                frame_position  = inner_start + int((inner_end - inner_start)/2)
                print (f"saving at frame {frame_position}")
                frame_to_save = cv2.resize(new_frames[b], (_smol_frame_width, _smol_frame_height), interpolation=cv2.INTER_LINEAR)
                bmethod.save_square_texture(frame_to_save, os.path.join(output_keys_folder, "keys{:05d}.png".format(frame_position)))
    
    just_frame_groups = []
    for i in range(len(bigprocessedbatches)):
        newgroup = []
        for b in range(len(bigprocessedbatches[i])):
            newgroup.append(bigprocessedbatches[i][b])
        just_frame_groups.append(newgroup)

    return

# keys_rel_dir 是相关的文件夹，用于对标输出文件的文件名，会从第一个开始对标，多了也不对标，少了也不对标，只按照顺序
# square_text_dir 存储gird图片的文件夹
# row_sides 多少行
# rol_sides 多少列

def split_square_images_to_singles(keys_rel_dir, row_sides, rol_sides, _smol_resolution, output_folder, square_textures):
    print(str(len(square_textures)) + "images to merge")
    texture_height, texture_width = square_textures[0].shape[:2]

    from os import walk

    f = []
    print("正在解析key_rel 目录")
    if len(keys_rel_dir) > 0:
        layer = 1
        w = walk(keys_rel_dir, topdown=False)  # 优先遍历顶层目录
        for (dirpath, dirnames, filenames) in w:
            if layer == 2:
                break
            for name in filenames:
                f.append(os.path.join(keys_rel_dir, name))
            layer += 1
    else:
        print("key_rel 目录为空，自动编号")
    file_list = sorted(f)

    last_index = 0
    for square_texture in square_textures:
        resized_square_texture = cv2.resize(square_texture, (texture_width, texture_height),
                                            interpolation=cv2.INTER_LINEAR)
        new_frames = bmethod.split_square_texture_2(resized_square_texture, row_sides, rol_sides,
                                            _smol_resolution)

        # 存储逻辑
        if os.path.exists(output_folder):
            os.remove(output_folder)
        os.makedirs(output_folder)
        for frame_to_save in new_frames:
            if last_index < len(file_list):
                frame_name = file_list[last_index]
            else:
                frame_name = str(last_index) + ".png"
            last_index += 1
            output_path = os.path.join(output_folder, frame_name)
            print(f"saving {output_path}")

            bmethod.save_square_texture(frame_to_save,
                                        output_path)


# def recombine (video_path, fps, per_side, batch_size, fillindenoise, edgedenoise, _smol_resolution,square_textures,max_frames,output_folder,border):
#     just_frame_groups = []
#     per_batch_limmit = (((per_side * per_side) * batch_size)) - border
#     video_data = bmethod.convert_video_to_bytes(video_path)
#     frames = bmethod.extract_frames_movpie(video_data, fps, max_frames)
#     bigbatches,frameLocs = bmethod.split_frames_into_big_batches(frames, per_batch_limmit,border,returnframe_locations=True)
#     bigprocessedbatches = []
#     for i in range(len(bigprocessedbatches)):
#         newgroup = []
#         for b in range(len(bigprocessedbatches[i])):
#             newgroup.append(bigprocessedbatches[i][b])
#         just_frame_groups.append(newgroup)
#
#     combined = bmethod.merge_image_batches(just_frame_groups, border)
#
#     save_loc = os.path.join(output_folder, "non_blended.mp4")
#     generated_vid = extensions.EbsyntHelper.scripts.berry_utility.pil_images_to_video(combined,save_loc, fps)




# def crossfade_folder_of_folders(output_folder, fps,return_generated_video_path=False):
#     """Crossfade between images in a folder of folders and save the results."""
#     root_folder = output_folder
#     all_dirs = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
#     dirs = [d for d in all_dirs if d.startswith("out_")]
#
#     dirs.sort()
#
#     output_images = []
#     allkeynums = getkeynums(os.path.join(root_folder, "keys"))
#     print(allkeynums)
#
#     for b in range(allkeynums[0]):
#         current_dir = os.path.join(root_folder, dirs[0])
#         images_current = sorted(os.listdir(current_dir))
#         image1_path = os.path.join(current_dir, images_current[b])
#         image1 = Image.open(image1_path)
#         output_images.append(np.array(image1))
#
#     for i in range(len(dirs) - 1):
#         current_dir = os.path.join(root_folder, dirs[i])
#         next_dir = os.path.join(root_folder, dirs[i + 1])
#
#         images_current = sorted(os.listdir(current_dir))
#         images_next = sorted(os.listdir(next_dir))
#
#         startnum = get_num_at_index(current_dir,0)
#         bigkeynum = allkeynums[i]
#         keynum = bigkeynum - startnum
#         print(f"recombining directory {dirs[i]} and {dirs[i+1]}, len {keynum}")
#
#
#
#
#         for j in range(keynum, len(images_current) - 1):
#             alpha = (j - keynum) / (len(images_current) - keynum)
#             image1_path = os.path.join(current_dir, images_current[j])
#             next_image_index = j - keynum if j - keynum < len(images_next) else len(images_next) - 1
#             image2_path = os.path.join(next_dir, images_next[next_image_index])
#
#             image1 = Image.open(image1_path)
#             image2 = Image.open(image2_path)
#
#             blended_image = butility.crossfade_images(image1, image2, alpha)
#             output_images.append(np.array(blended_image))
#             # blended_image.save(os.path.join(output_folder, f"{dirs[i]}_{dirs[i+1]}_crossfade_{j:04}.png"))
#
#     final_dir = os.path.join(root_folder, dirs[-1])
#     final_dir_images = sorted(os.listdir(final_dir))
#
#     # Find the index of the image with the last keyframe number in its name
#     last_keyframe_number = allkeynums[-1]
#     last_keyframe_index = None
#     for index, image_name in enumerate(final_dir_images):
#         number_in_name = int(''.join(filter(str.isdigit, image_name)))
#         if number_in_name == last_keyframe_number:
#             last_keyframe_index = index
#             break
#
#     if last_keyframe_index is not None:
#         print(f"going from dir {last_keyframe_number} to end at {len(final_dir_images)}")
#
#         # Iterate from the last keyframe number to the end
#         for c in range(last_keyframe_index, len(final_dir_images)):
#             image1_path = os.path.join(final_dir, final_dir_images[c])
#             image1 = Image.open(image1_path)
#             output_images.append(np.array(image1))
#     else:
#         print("Last keyframe not found in the final directory")
#
#
#     print (f"outputting {len(output_images)} images")
#     output_save_location = os.path.join(output_folder, "crossfade.mp4")
#     generated_vid = extensions.EbsyntHelper.scripts.berry_utility.pil_images_to_video(output_images, output_save_location, fps)
#
#     if return_generated_video_path == True:
#         return generated_vid
#     else:
#         return output_images

def getkeynums (folder_path):
    filenames = os.listdir(folder_path)

    # Filter filenames to keep only the ones starting with "keys" and ending with ".png"
    keys_filenames = [f for f in filenames if f.startswith("keys") and f.endswith(".png")]

    # Sort the filtered filenames
    sorted_keys_filenames = sorted(keys_filenames, key=lambda f: int(re.search(r'(\d+)', f).group(0)))

    # Extract the numbers from the sorted filenames
    return [int(re.search(r'(\d+)', f).group(0)) for f in sorted_keys_filenames]


def get_num_at_index(folder_path,index):
    """Get the starting number of the output images in a folder."""
    filenames = os.listdir(folder_path)

    # Filter filenames to keep only the ones starting with "keys" and ending with ".png"
    #keys_filenames = [f for f in filenames if f.startswith("keys") and f.endswith(".png")]

    # Sort the filtered filenames
    sorted_keys_filenames = sorted(filenames, key=lambda f: int(re.search(r'(\d+)', f).group(0)))

    # Extract the numbers from the sorted filenames
    numbers = [int(re.search(r'(\d+)', f).group(0)) for f in sorted_keys_filenames]
    return numbers[index]