#!/bin/zsh

if [ -z "$1" ]; then
  echo "사용법: $0 [대상_폴더]"
  exit 1
fi

target_dir="$1"

if [ ! -d "$target_dir" ]; then
  echo "에러: '$target_dir' 폴더가 존재하지 않습니다."
  exit 1
fi

for f in "$target_dir"/*.HEIC; do
  if [ -f "$f" ]; then
    echo "Working on file $f"
    heif-convert "$f" "$f".png
    rm "$f"
  fi
done
