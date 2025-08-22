import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml

from tagger import BlipTagger

# Supported image extensions for tagging
TAGGING_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

def run_tagging_phase(target_dir: Path, config_path: str = "tagger_config.yaml", device=None):
    print("\n" + "="*30)
    print("Starting BLIP Tagging Phase...")
    print("="*30)

    try:
        blip_tagger = BlipTagger(config_path=config_path, device=device)
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
        description="지정된 폴더의 이미지에 대해 BLIP 태그를 생성합니다.",
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

    target_path = Path(args.target_dir)
    
    if not target_path.is_dir():
        print(f"오류: '{args.target_dir}'는 유효한 디렉토리가 아닙니다.")
        return

    if args.gpu is None:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            print("오류: CUDA를 사용할 수 없습니다. GPU가 설치되어 있고 올바르게 구성되었는지 확인하세요.")
            return
        try:
            device = torch.device(f"cuda:{args.gpu}")
        except RuntimeError:
            print(f"오류: 지정된 GPU ID {args.gpu}를 찾을 수 없습니다.")
            return
    print(f"Using device: {device}")

    try:
        run_tagging_phase(target_path, device=device)

    except Exception as e:
        print(f"프로세스 초기화 중 치명적 오류 발생: {e}")

if __name__ == "__main__":
    main()
