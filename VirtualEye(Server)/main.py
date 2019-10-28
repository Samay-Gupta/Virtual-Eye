# Virtual Eye Server Code

from subfiles.connection import Server
from subfiles.interface import Display
from subfiles.classifier import Detector
from subfiles.functions import Handler
import time

clients = 1

if __name__ == "__main__":
    server_instance = Server()
    display_instance = Display()
    classifier = Detector('subfiles')
    classifier.train_dataset()
    #classifier.create_distance_dataset()
    classifier.load_dataset()
    host, port = server_instance.server_data['server_host']
    tag = ["Server: {}".format(host), "Port  : {}".format(port), ""]
    display_instance.add_permanent_tag(tag)
    thread_handlers = []
    for _ in range(clients):  
        client_id = server_instance.add_connection()
        handler = Handler(server_instance, display_instance, classifier, client_id)
        handler.start()
        thread_handlers.append(handler.main_thread)
    for thread_handler in thread_handlers:
        thread_handler.join()
    display_instance.write('All connections closed')
