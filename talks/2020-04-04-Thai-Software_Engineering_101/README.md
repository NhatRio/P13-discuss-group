# Installation
- Docker
- Docker-compose
- Linux, or WSL, or replace bash scripts in build/ with Windows scripts

# Build
```bash
cd build
bash -x build.sh
```

# Test

```bash
# POST users to user-service
curl -d '{"name":"Thi", "username": "thi_math"}' -H "Content-Type: application/json" -X POST http://localhost:5000/users

curl -d '{"name":"Thien", "username": "thien_math"}' -H "Content-Type: application/json" -X POST http://localhost:5000/users

# GET all users
curl http://localhost:5000/users

# GET one user
curl http://localhost:5000/users/thien_math

# POST channels to channel-service
curl -d '{"channel_id":"physics_101", "title": "Physics for Everyone"}' -H "Content-Type: application/json" -X POST http://localhost:5001/channels

curl -d '{"channel_id":"mathematics_101", "title": "Mathematics for Beginner"}' -H "Content-Type: application/json" -X POST http://localhost:5001/channels

# GET all channels
curl http://localhost:5001/channels

# GET one channel
curl http://localhost:5001/channels/mathematics_101

# Assign channel mathematics_101 to user thien_math
curl -H "Content-Type: application/json" -X POST http://localhost:5000/users/thien_math/channel/mathematics_101
```


# Troubleshooting

```bash
# Deploy one service in multiservice docker-compose.yaml
docker-compose up -d database

# Delete service container AND remove volume
docker-compose down database -v

# Connect to dockerized Mongo from localhost needs port:
mongo -u test localhost:27017
```