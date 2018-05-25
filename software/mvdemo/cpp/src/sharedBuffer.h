#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>

template<typename T>
class SharedBuffer
{
public:
    typedef T type;

    SharedBuffer(int capacity = 1)
    : m_front(0)
    , m_back(0)
    , m_capacity(capacity)
    , m_count(0)
    , m_buffer(capacity)
    {
    }

    int count()
    {
        return m_count;
    }

    void clear()
    {
        std::unique_lock<std::mutex> l(m_lock);
        m_front = 0;
        m_back = 0;
        m_count = 0;
        l.unlock();
        m_notEmpty.notify_one();
    }
    
    void put(const T& i_data)
    {
        std::unique_lock<std::mutex> l(m_lock);
        m_notFull.wait(l, [this](){return m_count != m_capacity;});
        m_buffer[m_back] = i_data;
        m_back = (m_back+1) % m_capacity;
        ++m_count;
        l.unlock();
        m_notEmpty.notify_one();
    }
    
    const T& get() const
    {
        return get();
    }
    
    T& get()
    {
        std::unique_lock<std::mutex> l(m_lock);
        m_notEmpty.wait(l, [this](){return m_count != 0;});
        
        T& res = m_buffer[m_front];
        m_front = (m_front + 1) % m_capacity;
        --m_count;
        
        l.unlock();
        m_notFull.notify_one();

        return res;
    }

private:
    int m_front;
    int m_back;
    int m_capacity;
    int m_count;
    
    std::vector<T> m_buffer;
    std::mutex m_lock;
    std::condition_variable m_notFull;
    std::condition_variable m_notEmpty;
};
