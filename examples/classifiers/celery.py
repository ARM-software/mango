from __future__ import absolute_import, unicode_literals
from celery import Celery

app = Celery('Mango',
             broker='amqp://',
             backend='rpc://',
             include=['mango.examples.classifiers.CeleryTasks'])


# Optional configuration
app.conf.update(
    result_expires=3600,
    broker_heartbeat = 0
)


if __name__ == '__main__':
    app.start()
