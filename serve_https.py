from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import ssl


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve VitaPro frontend over local HTTPS.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5500)
    parser.add_argument("--certfile", default="../certs/localhost.pem")
    parser.add_argument("--keyfile", default="../certs/localhost-key.pem")
    args = parser.parse_args()

    certfile = Path(args.certfile).resolve()
    keyfile = Path(args.keyfile).resolve()
    if not certfile.exists() or not keyfile.exists():
        raise SystemExit(
            f"Missing certificate files.\nCert: {certfile}\nKey: {keyfile}"
        )

    server = ThreadingHTTPServer((args.host, args.port), SimpleHTTPRequestHandler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print(f"Serving HTTPS on https://{args.host}:{args.port}/vita_final.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
