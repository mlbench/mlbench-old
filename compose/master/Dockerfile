FROM python:3.6-alpine

# Copy in your requirements file
ADD ./compose/master/requirements.txt /requirements.txt

# Install postgres
ENV PG_MAJOR 9.6
ENV PG_VERSION 9.6.8

ENV PATH /usr/lib/postgresql/$PG_MAJOR/bin:$PATH
ENV PGDATA /var/lib/postgresql/data

ENV LANG en_US.utf8

RUN apk update && apk add build-base readline-dev openssl-dev zlib-dev libxml2-dev glib-lang wget gnupg ca-certificates libssl1.0 && \
    gpg --keyserver ipv4.pool.sks-keyservers.net --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4 && \
    gpg --list-keys --fingerprint --with-colons | sed -E -n -e 's/^fpr:::::::::([0-9A-F]+):$/\1:6:/p' | gpg --import-ownertrust && \
    wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/1.7/gosu-amd64" && \
    wget -O /usr/local/bin/gosu.asc "https://github.com/tianon/gosu/releases/download/1.7/gosu-amd64.asc" && \
    gpg --verify /usr/local/bin/gosu.asc && \
    rm /usr/local/bin/gosu.asc && \
    chmod +x /usr/local/bin/gosu && \
    mkdir -p /docker-entrypoint-initdb.d && \
    wget https://ftp.postgresql.org/pub/source/v$PG_VERSION/postgresql-$PG_VERSION.tar.bz2 -O /tmp/postgresql-$PG_VERSION.tar.bz2 && \
    tar xvfj /tmp/postgresql-$PG_VERSION.tar.bz2 -C /tmp && \
    cd /tmp/postgresql-$PG_VERSION && ./configure --enable-integer-datetimes --enable-thread-safety --prefix=/usr/local --with-libedit-preferred --with-openssl  && make world && make install world && make -C contrib install && \
    cd /tmp/postgresql-$PG_VERSION/contrib && make && make install && \
    apk --purge del build-base openssl-dev zlib-dev libxml2-dev wget gnupg ca-certificates && \
    rm -r /tmp/postgresql-$PG_VERSION* /var/cache/apk/*

COPY ./compose/master/setup-postgres.sh /

# Install build deps, then run `pip install`, then remove unneeded build deps all in a single step. Correct the path to your production requirements file, if needed.
RUN set -ex \
    && apk add --no-cache --virtual .build-deps \
            gcc \
            make \
            libc-dev \
            musl-dev \
            linux-headers \
            pcre-dev \
            git \
    && python -m venv /venv \
    && /venv/bin/pip install -U pip \
    #&& git clone https://github.com/istrategylabs/django-rq-scheduler.git \
    #&& cd django-rq-scheduler \
    #&& LIBRARY_PATH=/lib:/usr/lib /bin/sh -c '/venv/bin/pip install --no-cache-dir . ' \
    #&& cd .. \
    && LIBRARY_PATH=/lib:/usr/lib /bin/sh -c "/venv/bin/pip install --no-cache-dir -r /requirements.txt" \
    && runDeps="$( \
            scanelf --needed --nobanner --recursive /venv \
                    | awk '{ gsub(/,/, "\nso:", $2); print "so:" $2 }' \
                    | sort -u \
                    | xargs -r apk info --installed \
                    | sort -u \
    )" \
    && apk add --virtual .python-rundeps $runDeps \
    && apk add --no-cache \
            nginx \
            sqlite \
            supervisor \
            redis \
            postgresql \
    && mkdir /data \
    && chown -R redis:redis /data\
    && echo -e "include /etc/redis-local.conf\n" >> /etc/redis.conf \
    && mkdir /run/nginx/ \
    && apk del .build-deps

RUN echo "daemon off;" >> /etc/nginx/nginx.conf
COPY ./compose/master/nginx-app.conf /etc/nginx/conf.d/default.conf
COPY ./compose/master/supervisord.conf /etc/supervisor/conf.d/

# Copy your application code to the container (make sure you create a .dockerignore file if any large files or directories should be excluded)
RUN mkdir /app/
RUN mkdir /app/code/
RUN mkdir /app/static/
RUN mkdir /app/media/

WORKDIR /app/

COPY ./compose/master/uwsgi.ini /app/
COPY ./compose/master/uwsgi_params /app/

EXPOSE 80

ADD ./mlbench/master/ /app/code/

RUN rm /app/code/master/settings.py \
    && mv /app/code/master/settings_production.py /app/code/master/settings.py

#Setup django
RUN /venv/bin/python code/manage.py collectstatic
RUN sh /setup-postgres.sh
RUN rm /setup-postgres.sh
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]