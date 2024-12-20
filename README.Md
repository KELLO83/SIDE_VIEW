

![버스 좌석 탐지 시스템](./git_image/image.png)

# 버스 좌석 탐지 시스템


## 프로젝트 개요

이 프로젝트는 버스 내부의 CCTV 영상을 분석하여 실시간으로 좌석 점유 상태를 파악하는 시스템입니다. 어안렌즈로 촬영된 CCTV 영상을 평면화하고 YOLO 기반 객체 탐지를 통해 승객을 감지하여 좌석 점유 여부를 판단합니다.

## 주요 기능

- 어안렌즈 영상 평면화 (Fisheye to Plane)
- YOLO 기반 승객 탐지
- 좌석 위치 매핑
- 실시간 좌석 상태 시각화

## SIDE VIEW

### 카메라 위치
- CAM0: 앞문 위 설치
- CAM2: 중간문 위 설치

### 좌석 구성
1. 일반 좌석
- ROW1 ~ ROW4: 각 열마다 좌/우 좌석
- 좌석 상태: {'left': bool, 'right': bool}

2. 측면 좌석
- seat9: 앞쪽 측면 좌석
- seat10: 뒤쪽 측면 좌석
- 좌석 상태: {'seat9': bool, 'seat10': bool}

### 좌석 매핑 좌표



## 시스템 구조

### 1. 이미지 전처리 (fisheye2plane.py)
- 어안렌즈 영상을 평면 영상으로 변환
- 회전 및 시점 조정 기능 제공

### 2. 객체 탐지 (detect.py)
- YOLO 모델을 사용한 승객 탐지
- 좌석 영역 기반 위치 분석
- NMS(Non-Maximum Suppression) 적용

### 3. 좌석 매핑 (area_collect.py)
- 탐지된 승객과 좌석 위치 매핑
- 점유 상태 데이터 수집

### 4. 시각화 (sit_recognition.py)
- 버스 좌석 배치도 기반 시각화
- 실시간 좌석 상태 표시


## 사용 방법

1. 데이터 준비
- CCTV 영상을 frames 폴더에 저장
- 각 카메라별로 하위 폴더 구성 (cam0/, cam2/)

2. 실행

3. 결과 확인
- `results/` 폴더에서 시각화 결과 확인
- 실시간 모니터링은 GUI 창을 통해 확인

## 시스템 요구사항

- Python 3.8+
- PyTorch 1.8+
- OpenCV 4.5+
- CUDA 지원 GPU (권장)
- 최소 8GB RAM

## 라이센스

This project is licensed under the MIT License - see the LICENSE file for details.

## 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

///
