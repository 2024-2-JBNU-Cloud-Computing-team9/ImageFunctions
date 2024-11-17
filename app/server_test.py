import requests
import json
import base64


# Flask 서버 URL
url = 'http://3.210.19.76:8080/images'

headers = {
    "Content-Type": "application/json",
    # "Content-Type": 'multipart/form-data'
    # "Authorization": "Bearer YOUR_ACCESS_TOKEN"  # 인증 토큰이 필요하다면 추가
}

content_image_path = "/Users/hyeonwoo/Downloads/ImageFunctions-main/AdaIN_style_transfer/input/content/avril.jpg"
style_image_path = "/Users/hyeonwoo/Downloads/ImageFunctions-main/AdaIN_style_transfer/input/style/asheville.jpg"


with open(content_image_path, "rb") as content_image_file, open(style_image_path, "rb") as style_image_file:
    content_image_base64 = base64.b64encode(content_image_file.read()).decode('utf-8')
    style_image_base64 = base64.b64encode(style_image_file.read()).decode('utf-8')
    
    
print(content_image_base64)
# 요청 데이터 생성
payload = {
    "content_image": content_image_base64,
    "style_image": style_image_base64,
    "alpha": 1
}
    

# POST 요청 전송
# response = requests.post(url, files=payload)

# response = requests.post(url, headers=headers, data = payload)
# # json
response = requests.post(url, headers=headers, data=json.dumps(payload))
# response = requests.post(url, headers=headers, files=payload)

# 응답 출력
if response.status_code == 200:
    response_data = response.json()
    print("Success:", response_data['message'])
else:
    print("Request failed with status code:", response.status_code)
    print("Error:", response.text)