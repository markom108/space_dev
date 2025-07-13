'''my server
USTAWIENIA servera:
host='127.0.0.1' #lokalny host
port=5000 #dowolny wolny port
'''
import socket #used for creating server


#PARAMETRY SERVERA:
host='127.0.0.1' #lokalny host
port=5000 #dowolny wolny port
MAX_CONECTIONS=1
MAX_RECEIVE=1024 #ile maksymalnie bajtów chcesz odebrać za jednym razem

server_socket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)#tworzę GNIAZDO serwera 
server_socket.bind((host,port))#BINDING - przypisz gniazdo do adresu i portu
server_socket.listen(MAX_CONECTIONS)# w nawiasie liczba maksymalnych oczekujących połączeń
    
#informacja dla użytkownika, że server nasłuchuje
print(f"Server running.Listening on {host}:{port}")
print("Waiting for a connection...")

#BLOKOWANIE - do póki nie połaczy z klientem, czeka
connection,adress=server_socket.accept() #nowy OBIEKT socketu(kanał) do komunikacji z konrkretnym klientem, krotka z adresem klienta
print(f"Connected by: {adress}")

#odbieranie danych
with connection: #dzięki temu nie trzeba zamykać  nowego socketu
    while True:
        data = connection.recv(MAX_RECEIVE)
        if not data:
            print("ERROR: 3")#problem z danymi
        else:
            print(data.decode())#to tekst przesłany w typie bytes stąd decode

server_socket.close()
print("Server closed.")


