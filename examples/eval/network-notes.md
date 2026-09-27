# Network notes

## Addresses

The router is the FritzBox 7590 from the internet provider, at 192.168.1.1. It hands out addresses between 192.168.1.100 and 192.168.1.199 by DHCP.

The server has a fixed address, 192.168.1.20, set in its netplan config rather than reserved on the router, so it keeps working even if the router is reset to factory settings.

## DNS

Pi-hole runs on the server and answers DNS for the whole house, blocking ads and trackers. Its upstream is Quad9 (9.9.9.9).

If websites stop loading on phones and laptops but the router's lights look normal, check the Pi-hole container first: when it's stopped, nothing can look up names. As a stopgap, switch the device's DNS to the router's address.

## Remote access

WireGuard listens on port 51820/UDP, which is the only port forwarded on the router. Nothing else is reachable from outside.

To add a device, run ./add-peer.sh with a name for it in /srv/compose/wireguard and scan the QR code it prints with the WireGuard app. Remove a device by deleting its folder under peers/ and restarting the container.

## Wi-Fi

The main network is "Hallway" (WPA3). Visitors use "Hallway Guests", which is isolated from the rest of the network and can only reach the internet.
