# 이미지 기능 모음

이 프로젝트는 다양한 이미지 처리 기능을 포함하고 있습니다.

## 1. 스타일 트랜스퍼
이 기능은 VGG 네트워크를 이용한 스타일 트랜스퍼 기능으로, 콘텐츠 이미지에 스타일 이미지를 적용하여 새로운 스타일의 이미지를 생성합니다.

### 사용법

1. **필수 파일 준비**:
   - `style_vgg` 관련 파일이 있어야 합니다. 특히 `model/style_vgg.py` 파일 내에 `make_net`, `GramMatrix`, `GramMSELoss`, `preprocess`, `postprocess` 함수 및 클래스가 정의되어 있어야 합니다.

2. **명령어 형식**:
   - 스타일 트랜스퍼를 실행하려면 아래와 같은 명령어를 터미널에서 실행합니다.
   ```bash
   python style_vgg.py --style_path <스타일_이미지_경로> --content_path <콘텐츠_이미지_경로> --output_path <결과_이미지_경로> --device <장치_옵션>

3. ***명령어 예시***:
   ```bash
   python <script_name>.py --style_path style.jpg --content_path content.jpg --output_path output.jpg --device cuda

## 2. Adain을 이용한 스타일 트랜스퍼

**출처**: [AdaIn_github](https://github.com/naoto0804/pytorch-AdaIN)

### 사용법

1. **필수 파일 준비**: Adain 스타일 트랜스퍼를 사용하기 위해 `vgg_normalised.pth`와 `decoder.pth` 또는 `decoder_iter_160000.pth.tar` 파일이 필요합니다. 이 파일들은 각각 VGG 네트워크와 디코더의 사전 학습된 가중치입니다.
   /
├── models/
│   ├── vgg.pth
│   └── decoder.pth
├── inputs/
│   ├── contents/
│   │   └── <콘텐츠_이미지_파일들>
│   └── styles/
│       └── <스타일_이미지_파일들>

   
2. **명령어 형식**:
   ```bash
   python test.py --content <콘텐츠_이미지_경로> --style <스타일_이미지_경로> --vgg models/vgg_normalised.pth --decoder models/decoder.pth --alpha <알파값>
