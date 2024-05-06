#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python3 server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 0 2); do
    echo "Starting client $i"
    python3 client.py -p "$i" -n 10 -s "0.0.0.0:8080" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Wait for all background processes to complete
wait