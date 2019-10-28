import threading
import json

class Handler:
    def __init__(self, server_instance, display_instance, classifier_instance, client_id):
        self.serv = server_instance
        self.disp = display_instance
        self.clf = classifier_instance
        self.client_id = client_id
        self.main_thread = threading.Thread(target=self.run)
        self.image_count = 0
        self.category = ''
        self.disp.write("S: Connected to client {}".format(client_id))

    def start(self):
        self.main_thread.start()

    def extract_object(self, text):
        text = text.lower()
        for obj_name in self.clf.database["synonyms"].keys():
            for name in self.clf.database["synonyms"][obj_name]:
                if name in text:
                    return (obj_name, name)
        else:
            return ("!NULL", 0)

    def run(self):
        running = True
        while running:
            client_command = self.serv.recv_txt(self.client_id, 1024)
            self.write("C: {}".format(client_command))
            modes = ["fnd", "ast", "vis"]
            identifiers = {
                "fnd":["find", "where"],
                "ast":["assist"],
                "vis":["visualise", "visualize"],
            }
            cat = ''
            for mode in modes:
                if cat != '':
                    break
                for identifier in identifiers[mode]:
                    if identifier in client_command:
                        cat = mode.upper()
                        break
            if cat == "FND":
                self.obj = self.extract_object(client_command)
                if self.obj == ("!NULL", 0):
                    msg = "Object not identifiable"
            else:
                self.obj = ("!NULL", 0)
                msg = "!NULL"
            if cat.lower() in modes:
                
                self.category = str(cat)
                self.serv.send_txt(self.client_id, cat)
                self.write("S: {}".format(cat))
                self.method_main()
            else:
                self.serv.send_txt(self.client_id, )
                
    def write(self, text):
        self.disp.write(text)

    def method_main(self):
        filename = 'subfiles\\images\\client_images\\img-{}.jpg'.format(self.client_id.replace(':',''))
        r = self.serv.recv_img(self.client_id, filename)
        self.write('S: Image file {} recieved'.format(filename.split('\\')[-1]))
        text = self.clf.compute(filename, self.category, self.obj)
        self.serv.send_txt(self.client_id, text)
        self.write(text)
        if self.category == 'AST':
            self.method_main()
        else:
            self.category = ''
