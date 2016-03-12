#!/usr/bin/python

import SimpleHTTPServer
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
from SocketServer import ThreadingMixIn
import SocketServer
import logging
import cgi
import sys
import shutil
import os
import time
import json

parent_app = None
directory = 'pics'

class WebService():
    def __init__(self):
        if os.path.exists(directory):
            shutil.rmtree(directory)                        
        os.makedirs(directory)

    def initialize(self, I, port=8080, app=None):
        global parent_app 
        parent_app = app
        #httpd = SocketServer.TCPServer(("", port), ServerHandler)
        server = ThreadedHTTPServer(('192.168.0.18', 8080), ServerHandler)
        
        
        print "Serving at: http://%s:%s" % (I , port)
        #httpd.serve_forever()
        server.serve_forever()
        
        
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass
        

class ServerHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        logging.warning("======= GET STARTED =======")
        logging.warning(self.headers)
        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        #logging.warning("======= POST STARTED =======")
        #logging.warning(self.headers)
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })
        #logging.warning("======= POST VALUES =======")
        
        #print 'Content-Length : %s' % self.headers['Content-Length']

        for item in form.list:
            if item.name == 'name':
                file_name = item.value
            elif item.name == 'mode':
                mode = item.value
            elif item.name == 'data':
                f = open(directory + '/' + file_name, 'wb+')
                f.write(item.value)
                f.close()

            #print "%s=%s" % (item.name, item.value)
        
        #print(self.rfile.read().decode("UTF-8"))
        
        """
        content = bytes("TEST RESPONSE").encode("UTF-8")
        self.send_response(200)
        self.send_header("Content-type","text/plain")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        
        #print(self.rfile.read().decode("UTF-8"))
        
        self.wfile.write(content)
        
        global parent_app
        if parent_app != None:
            parent_app.process_data(directory + '/' + file_name)
        """
            
        content = bytes("NULL RESPONSE").encode("UTF-8")
        
        global parent_app
        if parent_app != None:
            ret = parent_app.process_data(directory + '/' + file_name, mode)
            if ret != None:
                ret_str = json.dumps(ret)
                content = bytes(ret_str).encode("UTF-8")

        #print 'content : %s' % content
        
        self.send_response(200)
        self.send_header("Content-type","text/plain")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)
        
        
if __name__ == '__main__':
    WebService().initialize("192.168.0.18", 8080)
