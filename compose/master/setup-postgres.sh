echo "Fixing permissions on $PGDATA"
mkdir -p "$PGDATA"
chmod 0700 "$PGDATA"
chown -R postgres "$PGDATA"

echo "Fixing permissions on /run/postgresql"
mkdir -p /run/postgresql
chmod g+s /run/postgresql
chown -R postgres /run/postgresql

echo "Initializing database"
gosu postgres initdb

pass=
authMethod=trust

{ echo; echo "host all all 127.0.0.0/24 $authMethod"; } >> "$PGDATA/pg_hba.conf"
ps aux

echo "Creating database"
echo 'CREATE DATABASE mlbench;' | gosu postgres postgres --single -jE

echo "Starting database server"
gosu postgres pg_ctl -D "$PGDATA" \
        -w start
    : ${POSTGRES_USER:=postgres}
    : ${POSTGRES_DB:=$POSTGRES_USER}
    export POSTGRES_USER POSTGRES_DB
ps aux

#exec gosu postgres &
echo "Running django migrations"
/venv/bin/python code/manage.py migrate --noinput
/venv/bin/python code/manage.py flush --noinput
/venv/bin/python code/manage.py loaddata scheduled_tasks

gosu postgres pg_ctl -D "$PGDATA" -m fast -w stop