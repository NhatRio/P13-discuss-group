mongo <<EOF
var user = '$MONGO_INITDB_ROOT_USERNAME';
var passwd = '$MONGO_INITDB_ROOT_PASSWORD';
var admin = db.getSiblingDB('admin');
admin.auth(user, passwd);
var xpmdb = db.getSiblingDB('test');
xpmdb.createCollection('users');
xpmdb.createCollection('channels');
xpmdb.createUser({user:'test', pwd:'test@2020', roles: [{role:'readWrite', db: 'test'}]});
EOF
