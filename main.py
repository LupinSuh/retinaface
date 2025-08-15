import torch
import argparse
import time
import shutil
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from retinaface import RetinaFace
from tqdm import tqdm
import numpy as np

# 지원할 이미지 확장자 목록
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

def move_file_to(image_path: Path, subfolder_name: str):
    """지정된 하위 폴더로 파일을 이동시킵니다."""
    try:
        target_dir = image_path.parent / subfolder_name
        target_dir.mkdir(exist_ok=True)
        shutil.move(str(image_path), str(target_dir / image_path.name))
    except Exception as e:
        # 이동 중 오류가 발생하면 예외를 다시 발생시켜 상위 핸들러가 처리하도록 함
        raise e

def get_image_files(target_dir: Path) -> list[Path]:
    """
    대상 디렉토리와 그 하위 디렉토리에서 이미지 파일을 검색합니다.
    'Pass'와 'Fail' 폴더는 탐색에서 제외합니다.
    """
    image_files = []
    for root, dirs, files in os.walk(target_dir):
        # 'Pass'와 'Fail' 디렉토리는 더 이상 탐색하지 않도록 목록에서 제거
        dirs[:] = [d for d in dirs if d not in ['Pass', 'Fail']]
        for filename in files:
            if Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.append(Path(root) / filename)
    return image_files

def process_images(target_dir: Path):
    """
    이미지 파일을 분류하고 지정된 폴더로 이동시킵니다.
    """
    stats = {
        "Pass": 0,
        "LowRes": 0,
        "NoFace": 0,
        "MultiFace": 0,
        "NotIMG": 0,
        "Errors": 0
    }

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    try:
        model = RetinaFace
    except Exception as e:
        print(f"모델 로딩 중 치명적 오류가 발생했습니다: {e}")
        return

    try:
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = model.detect_faces(dummy_image)
    except Exception:
        pass

    image_files = get_image_files(target_dir)
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

                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        if width < 1024 or height < 1024:
                            stats["LowRes"] += 1
                            move_file_to(image_path, "Fail")
                            continue
                except UnidentifiedImageError:
                    stats["NotIMG"] += 1
                    move_file_to(image_path, "Fail")
                    continue
                except Exception as e:
                    log_bar.write(f"파일 처리 중 오류: {image_path}, {e}")
                    stats["Errors"] += 1
                    move_file_to(image_path, "Fail")
                    continue

                faces = model.detect_faces(str(image_path))
                num_faces = len(faces)

                if num_faces == 0:
                    stats["NoFace"] += 1
                    move_file_to(image_path, "Fail")
                elif num_faces == 1:
                    stats["Pass"] += 1
                    move_file_to(image_path, "Pass")
                else:
                    stats["MultiFace"] += 1
                    move_file_to(image_path, "Fail")

            except Exception as e:
                log_bar.write(f"이미지 처리 중 예기치 않은 오류 발생: {image_path}, {e}")
                stats["Errors"] += 1
                try:
                    move_file_to(image_path, "Fail")
                except Exception as move_e:
                    log_bar.write(f"Fail 폴더로 이동 중 추가 오류: {move_e}")
            finally:
                total_fails = stats['LowRes'] + stats['NoFace'] + stats['MultiFace'] + stats['NotIMG'] + stats['Errors']
                stats_str = f"✅ Pass: {stats['Pass']} | ❌ Fail: {total_fails} (LowRes:{stats['LowRes']}, NoFace:{stats['NoFace']}, MultiFace:{stats['MultiFace']}, NotIMG:{stats['NotIMG']}, Errors:{stats['Errors']})"
                stats_bar.set_description_str(stats_str)
                pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time
    print("\n")
    print("="*30)
    print("작업이 완료되었습니다.")
    print(f"총 소요 시간: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print("="*30)
    print("분류 결과 통계:")
    print(f"  - Pass: {stats['Pass']}개")
    print(f"  - Fail (Low Resolution): {stats['LowRes']}개")
    print(f"  - Fail (No Face Detected): {stats['NoFace']}개")
    print(f"  - Fail (Multiple Faces Detected): {stats['MultiFace']}개")
    print(f"  - Fail (Not an Image File): {stats['NotIMG']}개")
    if stats['Errors'] > 0:
        print(f"  - Errors (Processing Failed): {stats['Errors']}개")
    print("="*30)

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
        
    process_images(target_path)

if __name__ == "__main__":
    main()
