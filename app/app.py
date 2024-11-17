from flask import Flask, request, jsonify
import base64
import os
from datetime import datetime
import subprocess


app = Flask(__name__)

# 이미지 저장 경로 설정
# UPLOAD_FOLDER = "/home/ec2-user/uploads"
UPLOAD_FOLDER = '/home/ec2-user/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 처리된 이미지 경로 설정
RESULT_PATH = os.path.join(UPLOAD_FOLDER, "style.jpg")

# 결과 이미지 반환
@app.route('/processed-image', methods=['GET'])
def get_result_image():
    # 처리된 이미지가 존재하는지 확인
    if not os.path.exists(RESULT_PATH):
        return jsonify({"error": "Result image not found"}), 404

    # 이미지를 base64로 인코딩하여 JSON 형식으로 반환
    with open(RESULT_PATH, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
   
    return jsonify({"styled_image": encoded_image}), 200

# 폼 데이터를 통해 이미지 업로드
@app.route('/images', methods=['POST'])
def handle_image_upload():
    print(request.json)
    print(request.files)
    if "content_image" not in request.json or "style_image" not in request.json:
        # print(request.files)
        return jsonify({"error": "Both content and style images are required"}), 400

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S.jpg')
        print(timestamp)
        
        # 콘텐츠 이미지 저장
        content_file = request.json["content_image"]
        content_image_path = os.path.join(UPLOAD_FOLDER, "content", timestamp)
        
        content_data = content_file
        decoded_content_data = base64.b64decode(content_data)
        
        with open(content_image_path, "wb") as content_image:
            content_image.write(decoded_content_data)
        print('content saved')
        

        # 스타일 이미지 저장
        style_file = request.json["style_image"]
        style_image_path = os.path.join(UPLOAD_FOLDER, 'style', timestamp)
        
        # 파일 데이터 읽기 및 Base64 디코딩
        style_data = style_file
        decoded_style_data = base64.b64decode(style_data)
        
        # 디코딩된 데이터 저장
        with open(style_image_path, "wb") as style_image:
            style_image.write(decoded_style_data)
        print('style saved')
        
        # test.py 실행
        save_dir = os.path.join(UPLOAD_FOLDER, "output")
        vgg_path = '/home/ec2-user/ImageFunctions/AdaIN_style_transfer/models/vgg_normalised.pth'
        decoder_path = '/home/ec2-user/ImageFunctions/AdaIN_style_transfer/models/decoder.pth'
        # vgg_path = '/Users/hyeonwoo/Downloads/ImageFunctions-main/AdaIN_style_transfer/models/vgg_normalised.pth'
        # decoder_path = '/Users/hyeonwoo/Downloads/ImageFunctions-main/AdaIN_style_transfer/models/decoder.pth'
        alpha = request.json.get("alpha", 0.5)
        file_name = timestamp[:-4]
        print(file_name)
        
        command = [
            'python3', '/home/ec2-user/ImageFunctions/AdaIN_style_transfer/test.py',
            "--content", content_image_path,
            "--style", style_image_path,
            '--alpha', str(alpha),
            '--vgg', vgg_path,
            '--decoder', decoder_path,
            '--output', save_dir,
            '--output_file', file_name
        ]
        print(command)
    
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # 실행 결과 출력
        print("Command executed successfully")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        
        

        return jsonify({
            "message": "Images received and saved",
            "content_image_path": content_image_path,
            "style_image_path": style_image_path
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

"""@app.route('/processed-image', methods=['GET'])
def get_result_image():
    # 폼 데이터를 가져오기
    style_type = request.args.get('style_type')  # URL 쿼리 매개변수
    if not style_type:
        return jsonify({"error": "style_type parameter is required"}), 400

    # 처리된 이미지 파일 존재 여부 확인
    if not os.path.exists(RESULT_PATH):
        return jsonify({"error": "Result image not found"}), 404

    # 이미지 파일을 Base64로 인코딩
    with open(RESULT_PATH, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return jsonify({
        "styled_image": encoded_image,
        "style_type": style_type
    }), 200"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
