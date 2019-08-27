import json

with open('config_file_capture_image.json', 'r') as f:
    config = json.load(f)


#my_image=config['DEFAULT']['MAX_IMAGE_SIZE']
#my_num=config['DEFAULT']['MAX_IMAGE_SIZE_1']
#secret_key = config['DEFAULT']['SECRET_KEY'] # 'secret-key-of-myapp'
#ci_hook_url = config['CI']['HOOK_URL'] # 'web-hooking-url-from-ci-service'

image_type= config['DEFAULT']['image_type']
address= config['DEFAULT']['address']
print("type is:",type(image_type))
print(image_type)

print(type(address))