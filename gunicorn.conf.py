# gunicorn.conf.py
timeout = 120

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)
    # Dispose of the database engine created by the preload
    # so that each worker creates its own fresh connection.
    from app import app, db
    with app.app_context():
        db.engine.dispose()
