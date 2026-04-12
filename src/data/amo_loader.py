def load_amo_bench(n_problems=-1):
    logger.info("Loading AMO-Bench (meituan-longcat/AMO-Bench)")
    raw = load_dataset("meituan-longcat/AMO-Bench")
    split_name = "train" if "train" in raw else list(raw.keys())[0]
    data = list(raw[split_name])
    if n_problems > 0:
        data = data[:n_problems]
    problems = []
    for i, item in enumerate(data):
        question = item.get("prompt", "")
        answer = item.get("answer", "")
        answer_type = item.get("answer_type", "")
        problems.append({
            "problem_id": item.get("question_id", i),
            "question": str(question),
            "gold_answer": str(answer),
            "source": "amo_bench",
            "level": "olympiad",
            "problem_type": str(answer_type),
        })
    logger.info(f"Loaded {len(problems)} AMO-Bench problems")
    return problems