'''my server'''

import socket #used for creating server

    #tworzę GNIAZDO serwera
server_socket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    #BINDING - przypisz gniazdo do adresu i portu
host='127.0.0.1' #lokalny host
port=5000 #dowolny wolny port
server_socket.bind((host,port))

    #rozpocznij nasłuchiwanie:
server_socket.listen(1)# w nawiasie liczba maksymalnych oczekujących połączeń

    #informacja dla użytkownika, że server nasłuchuje
print(f"Serwer uruchomiony. Nasłuchiwanie na {host}:{port}")