
import sbibm

def get_tasks(name,**kwargs):
    sbibm_tasks = sbibm.get_available_tasks()
    if name in sbibm_tasks:
        task = sbibm.get_task(name, **kwargs)
    elif name == "pyloric":
        from sbivibm.tasks import Pyloric
        task = Pyloric(**kwargs)
    else:
        raise NotImplementedError("Unknown task")
    return task
