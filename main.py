import torch
import argparse
import time
import shutil
import os
from pathlib import Path
from tqdm import tqdm

from face import FaceDetector
from counter import Counter
from manager import FileManager
from tagger import BlipTagger # Import the new Tagger

# Supported image extensions for tagging
TAGGING_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

def process_images(target_dir: Path, file_manager: FileManager, face_detector: FaceDetector, counter: Counter):
    image_files = file_manager.get_image_files()
    if not image_files:
        print("처리할 이미지를 찾지 못했습니다.")
        return

    start_time = time.time()

    with tqdm(total=len(image_files), desc="Progress", unit="file", position=2, leave=False) as pbar, \
         tqdm(total=0, position=1, bar_format='{desc}') as stats_bar, \
         tqdm(total=0, position=0, bar_format='{desc}') as log_bar:
        
        stats_bar.set_description_str("✅ Pass: 0 | ❌ Fail: 0")

        for image_path in image_files:
            try:
                file_name = image_path.name
                terminal_width = shutil.get_terminal_size().columns
                max_len = terminal_width - 20
                if len(file_name) > max_len:
                    file_name = "..." + file_name[-(max_len-3):]
                
                log_bar.set_description_str(f"Processing: {file_name.ljust(terminal_width - 15)}")

                # Initial validation (is it a valid image file, resolution)
                validation_result = file_manager.validate_image(image_path)
                if validation_result != "Valid":
                    counter.increment(validation_result)
                    file_manager.move_file(image_path, "Fail")
                    continue

                # Process image for letterbox cropping and face detection
                faces, img_data, was_cropped = face_detector.process_image(str(image_path))
                if faces is None: # Error reading image in process_image
                    counter.increment("Errors")
                    file_manager.move_file(image_path, "Fail")
                    continue
                
                # Overwrite original file if it was cropped
                if was_cropped:
                    log_bar.write(f"  - Letterbox detected and cropped for: {file_name}")
                    file_manager.overwrite_image(img_data, image_path)

                # Determine face detection result
                num_faces = len(faces)
                if num_faces == 0:
                    face_result = "NoFace"
                elif num_faces == 1:
                    face_result = "Pass"
                else:
                    face_result = "MultiFace"
                
                counter.increment(face_result)

                # If the face result was a fail, move the (potentially cropped) file
                if face_result != "Pass":
                    file_manager.move_file(image_path, face_result)

            except Exception as e:
                log_bar.write(f"이미지 처리 중 예기치 않은 오류 발생: {image_path}, {e}")
                counter.increment("Errors")
                try:
                    file_manager.move_file(image_path, "Fail")
                except Exception as move_e:
                    log_bar.write(f"Fail 폴더로 이동 중 추가 오류: {move_e}")
            finally:
                stats = counter.get_stats()
                total_fails = counter.get_total_fails()
                stats_str = f"✅ Pass: {stats['Pass']} | ❌ Fail: {total_fails} (LowRes:{stats['LowRes']}, NoFace:{stats['NoFace']}, MultiFace:{stats['MultiFace']}, NotIMG:{stats['NotIMG']}, Errors:{stats['Errors']})"
                stats_bar.set_description_str(stats_str)
                pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time
    counter.set_total_time(total_time)
    print(counter.get_summary())

def run_tagging_phase(target_dir: Path, config_path: str = "tagger_config.yaml"):
    print("\n" + "="*30)
    print("Starting BLIP Tagging Phase...")
    print("="*30)

    try:
        blip_tagger = BlipTagger(config_path=config_path)
    except Exception as e:
        print(f"BLIP Tagger 초기화 중 오류 발생: {e}")
        return

    # List of (directory_path, prefix) tuples to process
    dirs_to_tag = []

    # Add the target_dir itself if it contains images
    # The prefix for images directly in target_dir will be its own name
    dirs_to_tag.append((target_dir, target_dir.name))

    # Add immediate subdirectories, excluding special ones
    for item in target_dir.iterdir():
        if item.is_dir() and item.name not in ['Fail', 'letterbox']:
            dirs_to_tag.append((item, item.name))
    
    if not dirs_to_tag:
        print("태그를 생성할 폴더를 찾지 못했습니다.")
        return

    total_images_to_tag = 0
    for dir_path, _ in dirs_to_tag:
        for file_path in dir_path.iterdir():
            if file_path.suffix.lower() in TAGGING_IMAGE_EXTENSIONS:
                total_images_to_tag += 1

    if total_images_to_tag == 0:
        print("태그를 생성할 이미지를 찾지 못했습니다.")
        return

    with tqdm(total=total_images_to_tag, desc="Tagging Progress", unit="file", position=2, leave=False) as pbar:
        for dir_path, prefix in dirs_to_tag:
            print(f"\nProcessing folder: {dir_path.name} (Prefix: {prefix})")
            for image_path in dir_path.iterdir():
                if image_path.suffix.lower() in TAGGING_IMAGE_EXTENSIONS:
                    try:
                        caption, error = blip_tagger.generate_tag(str(image_path))
                        if error:
                            print(f"Error tagging {image_path.name}: {error}")
                            continue

                        postfix = blip_tagger.get_postfix()
                        final_content = f"{prefix}, {caption}{postfix}" # Add comma after prefix

                        txt_file_path = image_path.with_suffix('.txt')
                        with open(txt_file_path, 'w', encoding='utf-8') as f:
                            f.write(final_content)
                        
                    except Exception as e:
                        print(f"Error processing {image_path.name} for tagging: {e}")
                    finally:
                        pbar.update(1)

    print("\n" + "="*30)
    print("BLIP Tagging Phase Completed.")
    print("="*30)

def main():
    parser = argparse.ArgumentParser(
        description="지정된 폴더의 이미지를 스캔하여 얼굴이 1개인 이미지를 'Pass' 폴더로 분류하고 BLIP 태그를 생성합니다.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "target_dir",
        type=str,
        help="이미지를 스캔할 대상 폴더 경로입니다."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="사용할 GPU의 ID를 지정합니다. (예: 0). 지정하지 않으면 CPU를 사용합니다."
    )
    
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    target_path = Path(args.target_dir)
    
    if not target_path.is_dir():
        print(f"오류: '{args.target_dir}'는 유효한 디렉토리가 아닙니다.")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    try:
        # Phase 1: Image Processing and Sorting
        face_detector = FaceDetector()
        file_manager = FileManager(target_path)
        counter = Counter()
        process_images(target_path, file_manager, face_detector, counter)

        # Phase 2: BLIP Tagging
        run_tagging_phase(target_path)

    except Exception as e:
        print(f"프로세스 초기화 중 치명적 오류 발생: {e}")

if __name__ == "__main__":
    main()
