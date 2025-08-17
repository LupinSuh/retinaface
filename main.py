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

                validation_result = file_manager.validate_image(image_path)
                if validation_result != "Valid":
                    counter.increment(validation_result)
                    file_manager.move_file(image_path, "Fail")
                    continue

                faces, _ = face_detector.detect_faces(str(image_path))
                num_faces = len(faces)

                if num_faces == 0:
                    result = "NoFace"
                elif num_faces == 1:
                    result = "Pass"
                else:
                    result = "MultiFace"
                
                counter.increment(result)
                file_manager.move_file(image_path, result)

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

def main():
    parser = argparse.ArgumentParser(
        description="지정된 폴더의 이미지를 스캔하여 얼굴이 1개인 이미지를 'Pass' 폴더로 분류합니다.",
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
        face_detector = FaceDetector()
        file_manager = FileManager(target_path)
        counter = Counter()
        process_images(target_path, file_manager, face_detector, counter)
    except Exception as e:
        print(f"프로세스 초기화 중 치명적 오류 발생: {e}")

if __name__ == "__main__":
    main()
