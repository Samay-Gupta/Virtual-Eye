import socket
import subprocess
import platform

class Server:
    def __init__(self):
        host = self.get_hostname()
        port = 8080
        self.server_data = {
            'server_host': (host, port),
            'active_connections': {},
        }
        self.server = socket.socket()
        self.id_count = 0
        self.server.bind(self.server_data['server_host'])

    def create_id(self):
        base_id = 'VE_CID:{}'
        id_len = 5
        self.id_count += 1
        id_val = (id_len-len(str(self.id_count)))*'0' + str(self.id_count)
        id_code = base_id.format(id_val)
        return id_code

    def get_hostname(self):
        network_command = 'ipconfig' if platform.system() == 'Windows' else 'ifconfig'
        command_result = subprocess.Popen(network_command, stdout=subprocess.PIPE).communicate()[0].decode()
        bind_attempt_order = ['192.168.', '169.154.', '10.0.']
        for ip_addr in bind_attempt_order:
            if ip_addr in command_result:
                ip_start_ind = command_result.index(ip_addr)
                ip_stop_ind = ip_start_ind + (command_result[ip_start_ind:]).index('\n')
                host_id = command_result[ip_start_ind:ip_stop_ind].strip()
                break
        else:
            host_id = socket.gethostname()
        return host_id

    def add_connection(self):
        self.server.listen(1)
        client_conn, (client_ip, client_port) = self.server.accept()
        conn_id = self.create_id()
        connection_data = {
            'conn': client_conn,
            'addr':{
                'ip': client_ip,
                'port': client_port,
            },
            'name': ''
        }
        self.server_data['active_connections'][conn_id] = connection_data
        self.server_data['active_connections'][conn_id]['name'] = self.recv_txt(conn_id)
        return conn_id
        
    def send_txt(self, conn_id, text=''):
        text = str(text) + '\r\n'
        conn_instance = self.server_data["active_connections"][conn_id]['conn']
        conn_instance.send(str(text).encode('UTF-8'))

    def recv_txt(self, conn_id, size=128):
        conn_instance = self.server_data["active_connections"][conn_id]['conn']
        text = (conn_instance.recv(size)[2:]).decode().strip()
        return text

    def send_img(self, conn_id, file='', packet_size=4096):
        conn_instance = self.server_data["active_connections"][conn_id]['conn']
        with open(file, 'rb') as img_file_object:
            file_data = img_file_object.read()
        file_size = len(file_data)
        text = "IMG:{}:{}".format(file_size, packet_size)
        conn_instance.send(str(text).encode('UTF-8'))
        conn_instance.recv(b'1')
        for pointer in range(0, file_size, packet_size):
            packet = file_data[:packet_size]
            conn_instance.send(packet)
            file_data = file_data[packet_size:]
        else:
            packet = file_data[:packet_size]
            conn_instance.send(packet)

    def recv_img(self, conn_id, file=''):
        conn_instance = self.server_data["active_connections"][conn_id]['conn']
        file_category, *res = (conn_instance.recv(128).decode()[2:]).split(':')
        file_size, packet_size = map(int, res)
        conn_instance.send(b'1\r\n')
        if file_category != "IMG":
            return (0, "Invalid File Category")
        else:
            file_data = b''
            for _ in range(0, file_size, packet_size):
                file_data += conn_instance.recv(packet_size)
                conn_instance.send(b'1\r\n')
            else:
                file_data += conn_instance.recv(file_size%packet_size)
            with open(file, 'wb') as img_file_object:
                img_file_object.write(file_data)
            return (1, "Image File saved")
            