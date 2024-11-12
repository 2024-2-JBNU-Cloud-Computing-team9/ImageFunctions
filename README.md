# 이미지 기능 모음

이 프로젝트는 다양한 이미지 처리 기능을 포함하고 있습니다.

## 1. 스타일 트랜스퍼


## 2. Adain을 이용한 스타일 트랜스퍼

이 기능은 Adaptive Instance Normalization (Adain) 기법을 이용하여 콘텐츠 이미지에 스타일 이미지를 적용하는 스타일 트랜스퍼 기능입니다. PyTorch를 기반으로 구현되었으며, 다양한 설정을 통해 스타일 적용 정도와 콘텐츠 보존 비율을 조절할 수 있습니다.

### 사용법

1. **필수 파일 준비**: Adain 스타일 트랜스퍼를 사용하기 위해 `vgg_normalised.pth`와 `decoder.pth` 또는 `decoder_iter_160000.pth.tar` 파일이 필요합니다. 이 파일들은 각각 VGG 네트워크와 디코더의 사전 학습된 가중치입니다.
   
2. **명령어 형식**:
   ```bash
   python test.py --content <콘텐츠_이미지_경로> --style <스타일_이미지_경로> --vgg models/vgg_normalised.pth --decoder models/decoder.pth --alpha <알파값>
