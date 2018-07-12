FROM python:3.6-alpine

# Copy in your requirements file
ADD ./compose/master/requirements.txt /requirements.txt

# OR, if you’re using a directory for your requirements, copy everything (comment out the above and uncomment this if so):
# ADD requirements /requirements

# Install build deps, then run `pip install`, then remove unneeded build deps all in a single step. Correct the path to your production requirements file, if needed.
RUN set -ex \
    && apk add --no-cache --virtual .build-deps \
            gcc \
            make \
            libc-dev \
            musl-dev \
            linux-headers \
            pcre-dev \
    && python -m venv /venv \
    && /venv/bin/pip install -U pip \
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

#Setup django
RUN /venv/bin/python code/manage.py collectstatic
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]