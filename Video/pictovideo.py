import cv2
import os


def images_to_video(image_folder, video_name, frame_rate):


    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    print(f"Found {len(images)} images in the folder.")


    # images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))


    frame = cv2.imread(os.path.join(image_folder, images[0]))


    height, width, layers = frame.shape


    fourcc = cv2.VideoWriter_fourcc(*'DIVX')


    video = cv2.VideoWriter(video_name, fourcc, frame_rate, (width, height))


    for image in images:


        frame = cv2.imread(os.path.join(image_folder, image))


        video.write(frame)


    cv2.destroyAllWindows()


    video.release()

if __name__ == "__main__":
    image_folder = r'图片路径'
    video_name = r'视频路径'
    frame_rate = 30  # Adjust the frame rate as needed

    images_to_video(image_folder, video_name, frame_rate)
    print(f"Video saved as {video_name}")