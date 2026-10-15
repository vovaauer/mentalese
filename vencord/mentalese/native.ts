/*
 * Runs in Vesktop's main process: forwards packets to the mentalese app over local UDP.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

import { createSocket, Socket } from "dgram";

const PORT = 47321;
let socket: Socket | null = null;

export function send(_, data: Uint8Array) {
    if (!socket) {
        socket = createSocket("udp4");
        socket.on("error", () => { }); // mentalese not running: drop packets silently
    }
    socket.send(data, PORT, "127.0.0.1");
}

export function close() {
    socket?.close();
    socket = null;
}
