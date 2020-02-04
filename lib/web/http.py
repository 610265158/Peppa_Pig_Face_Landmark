from flask import Flask,request,Response
import json

from lib.core.api.facer import FaceAna
from lib.core.headpose.pose import get_head_pose, line_pairs

from lib.web.utils import parse_img,parse_as_dict


##init a model there as global model
facer = FaceAna()

app = Flask("peppa_pig_face_engine")


@app.route('/peppa_pig_face_engine/healthcheck')
def healthcheck():
    print('peppa_pig_face_engine/healthcheck excuted ')
    return 'Hello, i am peppa_pig_face'



@app.route('/peppa_pig_face_engine/excuter',methods=['GET','POST'])
def excuter():

    image=parse_img(request.data)

    boxes, landmarks, states = facer.run(image)

    ### no track
    facer.reset()

    res=parse_as_dict(landmarks)
    print('peppa_pig_face_engine/excuter,    one image processed ')
    print(res)
    response = Response(
        response=json.dumps(res),
        status=200,
        mimetype='application/json'
    )
    return response



if __name__ =='__main__':

    app.run(host='0.0.0.0',port=5000,  debug=True)

