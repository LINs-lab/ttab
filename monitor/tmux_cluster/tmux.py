# -*- coding: utf-8 -*-
from monitor.tmux_cluster.utils import ossystem
import shlex

TASKDIR_PREFIX = "/tmp/tasklogs"


def exec_on_node(cmds, host="localhost"):
    def _decide_node(cmd):
        return cmd if host == "localhost" else f"ssh {host} -t {shlex.quote(cmd)}"

    cmds = (
        [_decide_node(cmd) for cmd in cmds]
        if isinstance(cmds, list)
        else _decide_node(cmds)
    )
    ossystem(cmds)


class Run(object):
    def __init__(self, name, job_node="localhost"):
        self.name = name
        self.jobs = []
        self.job_node = job_node

    def make_job(self, job_name, task_scripts, run=True, **kwargs):
        num_tasks = len(task_scripts)
        assert num_tasks > 0

        if kwargs:
            print("Warning: unused kwargs", kwargs)

        # Creating cmds
        cmds = []
        session_name = self.name + "-" + job_name  # tmux can't use . in name
        cmds.append(f"tmux kill-session -t {session_name}")

        windows = []
        for task_id in range(num_tasks):
            if task_id == 0:
                cmds.append(f"tmux new-session -s {session_name} -n {task_id} -d")
            else:
                cmds.append(f"tmux new-window -t {session_name} -n {task_id}")
            windows.append(f"{session_name}:{task_id}")

        job = Job(self, job_name, windows, task_scripts, self.job_node)
        job.make_tasks()
        self.jobs.append(job)
        if run:
            for job in self.jobs:
                cmds += job.cmds
            exec_on_node(cmds, self.job_node)
        return job

    def attach_job(self):
        raise NotImplementedError

    def kill_jobs(self):
        cmds = []
        for job in self.jobs:
            session_name = self.name + "-" + job.name
            cmds.append(f"tmux kill-session -t {session_name}")
        exec_on_node(cmds, self.job_node)


class Job(object):
    def __init__(self, run, name, windows, task_scripts, job_node):
        self._run = run
        self.name = name
        self.job_node = job_node
        self.windows = windows
        self.task_scripts = task_scripts
        self.tasks = []

    def make_tasks(self):
        for task_id, (window, script) in enumerate(
            zip(self.windows, self.task_scripts)
        ):
            self.tasks.append(
                Task(
                    window,
                    self,
                    task_id,
                    install_script=script,
                    task_node=self.job_node,
                )
            )

    def attach_tasks(self):
        raise NotImplementedError

    @property
    def cmds(self):
        output = []
        for task in self.tasks:
            output += task.cmds
        return output


class Task(object):
    """Local tasks interact with tmux session.

    * session name is derived from job name, and window names are task ids.
    * no pane is used.

    """

    def __init__(self, window, job, task_id, install_script, task_node):
        self.window = window
        self.job = job
        self.id = task_id
        self.install_script = install_script
        self.task_node = task_node

        # Path
        self.cmds = []
        self._run_counter = 0

        for line in install_script.split("\n"):
            self.run(line)

    def run(self, cmd):
        self._run_counter += 1

        cmd = cmd.strip()
        if not cmd or cmd.startswith("#"):
            # ignore empty command lines
            # ignore commented out lines
            return

        modified_cmd = cmd
        self.cmds.append(
            f"tmux send-keys -t {self.window} {shlex.quote(modified_cmd)} Enter"
        )

    def upload(self, source_fn, target_fn="."):
        raise NotImplementedError()

    def download(self, source_fn, target_fn="."):
        raise NotImplementedError()
