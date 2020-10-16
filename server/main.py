from enum import Enum
import logging
import numpy as np
import cv2
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
from gabriel_server import local_engine
import logging
import knex_pb2
import base64
import json
import requests
import os


THRESHOLD = 0.6
NOSE = 'nose'
FUSELAGE = 'fuselage'
REAR = 'rear'
TAIL = 'tail'
WINGS = 'wings'

# Max image width and height
IMAGE_MAX_WH = 640

IMAGE_DIR = 'images'


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _img_is_label(img, label):
    _, jpeg_frame = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(jpeg_frame).decode('utf-8')

    instances = {
            'instances': [
                    {'image_bytes': {'b64': str(encoded_image)},
                     'key': 'key'}
            ]
    }

    url = 'http://localhost:8501/v1/models/default:predict'  # change this??
    response = requests.post(url, data=json.dumps(instances))
    predictions = response.json()['predictions'][0]

    for predicted_label, score in zip(
            predictions['labels'], predictions['scores']):
        if predicted_label == label:
            logger.info('Label: %s\t Score: %f', predicted_label, score)
            return score > THRESHOLD

    raise Exception('Bad label {}'.format(label))


def _result_wrapper_for_state(state):
    status = gabriel_pb2.ResultWrapper.Status.SUCCESS
    result_wrapper = cognitive_engine.create_result_wrapper(status)

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.TEXT
    result.payload = state.get_speech().encode()
    result_wrapper.results.append(result)

    result = gabriel_pb2.ResultWrapper.Result()
    result.payload_type = gabriel_pb2.PayloadType.IMAGE
    result.payload = state.get_image_bytes()
    result_wrapper.results.append(result)

    logger.info('sending %s', state.get_speech())

    return result_wrapper


class KnexEngine(cognitive_engine.Engine):
    def __init__(self):
        self._state = State.NOTHING

    def handle(self, input_frame):
        if self._state == State.DONE:
            status = gabriel_pb2.ResultWrapper.Status.SUCCESS
            return cognitive_engine.create_result_wrapper(status)

        to_server = cognitive_engine.unpack_extras(knex_pb2.ToServer,
                                                   input_frame)

        assert len(input_frame.payloads) == 1
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            status = gabriel_pb2.ResultWrapper.Status.WRONG_INPUT_FORMAT
            return cognitive_engine.create_result_wrapper(status)

        width = to_server.width
        height = to_server.height
        if width > IMAGE_MAX_WH or height > IMAGE_MAX_WH:
            raise Exception('Image too large')

        yuv = np.frombuffer(input_frame.payloads[0], dtype=np.uint8)
        yuv = np.reshape(yuv, ((height + (height//2)), width))
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        img = np.rot90(img, 3)

        if self._state == State.NOTHING:
            return self._update_state_gen_response(State.NOSE)
        elif self._state == State.NOSE:
            if _img_is_label(img, NOSE):
                return self._update_state_gen_response(State.FUSELAGE)
        elif self._state == State.FUSELAGE:
            if _img_is_label(img, FUSELAGE):
                return self._update_state_gen_response(State.REAR)
        elif self._state == State.REAR:
            if _img_is_label(img, REAR):
                return self._update_state_gen_response(State.TAIL)
        elif self._state == State.TAIL:
            if _img_is_label(img, TAIL):
                return self._update_state_gen_response(State.WINGS)
        elif self._state == State.WINGS:
            if _img_is_label(img, WINGS):
                return self._update_state_gen_response(State.DONE)
        else:
            raise Exception('Bad State')

        status = gabriel_pb2.ResultWrapper.Status.SUCCESS
        return cognitive_engine.create_result_wrapper(status)

    def _update_state_gen_response(self, state):
        self._state = state
        return _result_wrapper_for_state(state)


class State(Enum):
    NOTHING = (None, None)
    NOSE = ('Build the nose and put it on the table.', 'nose.jpg')
    FUSELAGE = ('Now build the fuselage and attach it to the back of the nose.', 'fuselage.jpg')
    REAR = ('Now build the rear section and attach it to the back of the fuselage.', 'rear.jpg')
    TAIL = ('Now build the tail and attach it to the top of the rear section.', 'tail.jpg')
    WINGS = ('Now build the wings and attach them to the sides of the fuselage.', 'wings.jpg')
    DONE = ('You are done! Happy flying.', 'wings.jpg')

    def __init__(self, speech, image_filename):
        self._speech = speech
        if image_filename is not None:
            image_path = os.path.join(IMAGE_DIR, image_filename)
            self._image_bytes = open(image_path, 'rb').read()

    def get_speech(self):
        return self._speech

    def get_image_bytes(self):
        return self._image_bytes


def main():
    def engine_factory():
        return KnexEngine()

    local_engine.run(engine_factory, 'knex', 60, 9099, 2)


if __name__ == '__main__':
    main()
