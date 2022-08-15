import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from typing import Tuple
import numpy as np
import tensorflow as tf


class request(BaseHTTPRequestHandler):

    def do_POST(self):
        # 获取请求体的长度去读取文件
        content_length = int(self.headers['Content-Length'])
        post_data = json.loads(self.rfile.read(content_length))
        print(post_data)
        print(self.headers['Content-Length'])
        print(post_data['input'])
        send_data = (np.array(((model(tf.reshape([float(post_data['input'])], (1, 1)))))) - 1).tolist()

        print(send_data)
        # send_data = {"name": 'woc', "age": 18}
        # send_data.update({'gender': 'nan'})
        send = {}
        send.update({'output': send_data})
        send_data = json.dumps(send)
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(send_data)))
        self.send_header('Access-Control-Allow-Origin', "*")
        self.end_headers()

        # 发送数据，数据正好是比特率
        self.wfile.write(bytearray(send_data, 'utf-8'))

    def do_GET(self):
        self.send_response(200)
        # self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
        self.send_header('Cache-Control', 'max-age=3000')
        self.send_header('Content-Length', '20')
        self.end_headers()
        self.wfile.write(b'console.log(\'here\');')

    # 不让python的控制台打印log信息
    def log_message(self, format, *args):
        return


def main():
    ts = HTTPServer(('127.0.0.1', 8999), request)
    ts.serve_forever()


if __name__ == '__main__':
    global model
    model = tf.saved_model.load('./savedModel')
    main()
