from enum import Enum
import logging
import numpy as np
import cv2
from gabriel_server import cognitive_engine
from gabriel_protocol import gabriel_pb2
from gabriel_server import local_engine
import logging
import sandwich_pb2
import base64
import json
import requests
import os


THRESHOLD = 0.6
BREAD = 'bread'
HAM = 'ham'
LETTUCE = 'lettuce'
HALF = 'half'
TOMATO = 'tomato'
FULL = 'full'

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

    url = 'http://localhost:8501/v1/models/default:predict'
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


class SandwichEngine(cognitive_engine.Engine):
    def __init__(self):
        self._state = State.NOTHING

    def handle(self, input_frame):
        if self._state == State.DONE:
            status = gabriel_pb2.ResultWrapper.Status.SUCCESS
            return cognitive_engine.create_result_wrapper(status)

        to_server = cognitive_engine.unpack_extras(sandwich_pb2.ToServer,
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
            return self._update_state_gen_response(State.BREAD)
        elif self._state == State.BREAD:
            if _img_is_label(img, BREAD):
                return self._update_state_gen_response(State.HAM)
        elif self._state == State.HAM:
            if _img_is_label(img, HAM):
                return self._update_state_gen_response(State.LETTUCE)
        elif self._state == State.LETTUCE:
            if _img_is_label(img, LETTUCE):
                return self._update_state_gen_response(State.HALF)
        elif self._state == State.HALF:
            if _img_is_label(img, HALF):
                return self._update_state_gen_response(State.TOMATO)
        elif self._state == State.TOMATO:
            if _img_is_label(img, TOMATO):
                return self._update_state_gen_response(State.FULL)
        elif self._state == State.FULL:
            if _img_is_label(img, FULL):
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
    BREAD = ('My name is Barry B Benson.', 'bread.jpg')
    HAM = ('Now put a piece of ham on the bread.', 'ham.jpg')
    LETTUCE = ('Now put a piece of lettuce on the ham.', 'lettuce.jpg')
    HALF = ('Now put a piece of bread on the lettuce.', 'half.jpg')
    TOMATO = ('Now put a piece of tomato on the bread.', 'tomato.jpg')
    FULL = ('Now put the bread on top.', 'full.jpg')
    DONE = ('You are done!', 'full.jpg')

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
        return SandwichEngine()

    local_engine.run(engine_factory, 'sandwich', 60, 9099, 2)


if __name__ == '__main__':
    main()
