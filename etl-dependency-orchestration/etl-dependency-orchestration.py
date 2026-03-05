def schedule_pipeline(tasks, resource_budget):
    """
    Schedules ETL tasks using a greedy approach respecting dependencies and resources.
    """
    # Initialize task state tracking
    task_lookup = {t['name']: t for t in tasks}
    completed_tasks = set()
    running_tasks = [] # List of (end_time, task_name)
    scheduled_times = {} # task_name -> start_time
    
    # Track which tasks are still waiting to be started
    pending_tasks = set(t['name'] for t in tasks)
    current_time = 0
    
    while pending_tasks or running_tasks:
        # 1. Complete tasks that have finished by current_time
        still_running = []
        for end_time, name in running_tasks:
            if end_time <= current_time:
                completed_tasks.add(name)
            else:
                still_running.append((end_time, name))
        running_tasks = still_running

        # 2. Identify ready tasks (dependencies met, not started)
        ready_tasks = []
        for name in pending_tasks:
            task = task_lookup[name]
            if all(dep in completed_tasks for dep in task['depends_on']):
                ready_tasks.append(name)
        
        # 3. Sort ready tasks alphabetically to break ties
        ready_tasks.sort()

        # 4. Greedily assign tasks based on resource budget
        current_resources = sum(task_lookup[name]['resources'] for _, name in running_tasks)
        
        tasks_to_remove_from_pending = []
        for name in ready_tasks:
            task = task_lookup[name]
            if current_resources + task['resources'] <= resource_budget:
                # Start the task
                scheduled_times[name] = current_time
                running_tasks.append((current_time + task['duration'], name))
                current_resources += task['resources']
                tasks_to_remove_from_pending.append(name)
        
        for name in tasks_to_remove_from_pending:
            pending_tasks.remove(name)

        # 5. Advance time to the next task completion
        if running_tasks:
            current_time = min(end_time for end_time, _ in running_tasks)
        elif pending_tasks:
            # This case shouldn't be reachable in a valid DAG where tasks can fit the budget
            break

    # Format and sort results by (start_time, name)
    result = [(name, start) for name, start in scheduled_times.items()]
    return sorted(result, key=lambda x: (x[1], x[0]))