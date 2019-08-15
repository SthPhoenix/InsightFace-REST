FROM tensorflow-opencv:preconf

WORKDIR /app

COPY ./src /app/src

ENTRYPOINT [ "python" ]

CMD [ "src/api/app.py" ]
