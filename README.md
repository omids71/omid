# omid
import numpy as np
import gym

# ایجاد محیط FrozenLake
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# تعداد حالات و اعمال
state_space = env.observation_space.n
action_space = env.action_space.n

# ایجاد جدول Q با مقادیر اولیه صفر
q_table = np.zeros((state_space, action_space))

# پارامترهای یادگیری
alpha = 0.1  # نرخ یادگیری
gamma = 0.99  # ضریب تخفیف
epsilon = 1.0  # مقدار اولیه اکتشاف
epsilon_decay = 0.99  # کاهش مقدار اکتشاف
min_epsilon = 0.1  # حداقل مقدار اکتشاف
episodes = 1000  # تعداد اپیزودها

# حلقه یادگیری
for episode in range(episodes):
    state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)  # بازنشانی محیط
    state = int(state)  # اطمینان از اینکه state یک عدد صحیح است
    done = False

    while not done:
        # انتخاب عمل با استفاده از سیاست ε-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # انتخاب تصادفی
        else:
            action = np.argmax(q_table[state])  # انتخاب بهترین عمل

        # اجرای عمل و دریافت پاداش
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = int(next_state)  # اطمینان از اینکه next_state یک عدد صحیح است
        done = terminated or truncated

        # به‌روزرسانی جدول Q
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        # انتقال به حالت بعدی
        state = next_state

    # کاهش مقدار ε
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# نمایش جدول Q نهایی
print("Training finished.\nFinal Q-Table:")
print(q_table)

# آزمایش سیاست بهینه
state, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), None)
state = int(state)  # اطمینان از اینکه state یک عدد صحیح است
env.render()
done = False
while not done:
    action = np.argmax(q_table[state])
    next_state, _, terminated, truncated, _ = env.step(action)
    next_state = int(next_state)  # اطمینان از اینکه next_state یک عدد صحیح است
    done = terminated or truncated
    state = next_state
    env.render()
