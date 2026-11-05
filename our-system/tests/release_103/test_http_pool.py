from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timedelta,timezone
from ipaddress import ip_address
import json,ssl,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import httpx,pytest
from app.embedding_http_pool import TransportPool
from app.embeddings import EmbeddingConfig,HttpEmbeddingProvider

def test_client_lease_does_not_close_shared_transport_or_keep_cookies(monkeypatch):
    closed=[];cookies=[]
    class Transport(httpx.MockTransport):
        def close(self):closed.append(True)
    def handler(request):
        cookies.append(request.headers.get('cookie'))
        return httpx.Response(200,json={'data':[{'index':0,'embedding':[1,0]}]},headers={'Set-Cookie':'private=synthetic'})
    provider=HttpEmbeddingProvider(EmbeddingConfig(provider='engineering',endpoint='https://invalid.test/embed',dimensions=2),transport=Transport(handler))
    monkeypatch.setenv('AML_EMBEDDING_API_KEY','synthetic')
    with ThreadPoolExecutor(16) as pool:
        batches=list(pool.map(lambda _:provider.embed(['synthetic']),range(32)))
    assert all(b.vectors==[[1.,0.]] for b in batches)
    assert cookies==[None]*32 and not closed
    provider._http_pool.close();provider._http_pool.close()
    assert closed==[True]

def test_real_local_https_reuses_one_connection_for_separate_embeds(tmp_path,monkeypatch):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    subject=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'synthetic-local')])
    now=datetime.now(timezone.utc)
    cert=(x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1))
        .not_valid_after(now+timedelta(hours=1)).add_extension(x509.SubjectAlternativeName([x509.IPAddress(ip_address('127.0.0.1'))]),critical=False)
        .sign(key,hashes.SHA256()))
    cp=tmp_path/'cert.pem';kp=tmp_path/'key.pem'
    cp.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    kp.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    peers=[];cookies=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*a):pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            peers.append(self.client_address);cookies.append(self.headers.get('Cookie'))
            data=json.dumps({'data':[{'index':0,'embedding':[1,0]}]}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(data)))
            self.send_header('Set-Cookie','secret=synthetic');self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(cp,kp)
    server.socket=context.wrap_socket(server.socket,server_side=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    transport=httpx.HTTPTransport(verify=ssl.create_default_context(cafile=str(cp)),trust_env=False,retries=0)
    provider=HttpEmbeddingProvider(EmbeddingConfig(provider='engineering',endpoint='https://127.0.0.1:'+str(server.server_port)+'/embed',dimensions=2),transport=transport)
    monkeypatch.setenv('AML_EMBEDDING_API_KEY','synthetic')
    try:
        for _ in range(5):assert provider.embed(['synthetic']).vectors==[[1.,0.]]
        assert len(peers)==5 and len(set(peers))==1 and cookies==[None]*5
    finally:
        provider._http_pool.close();server.shutdown();server.server_close();thread.join(timeout=2)
