#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

template <size_t Bits>
class RegisterN {
    static_assert(Bits % 64 == 0, "Register size must be multiple of 64");

public:
    static constexpr size_t WordBits  = 64;
    static constexpr size_t WordCount = Bits / WordBits;

private:
    std::array<uint64_t, WordCount> w{};

public:
    // ===== Constructors =====
    RegisterN() { clear(); }

    explicit RegisterN(uint64_t value) {
        clear();
        w[0] = value;
    }

    // ===== Basic operations =====
    void clear() { w.fill(0); }

    bool isZero()  {
        for (auto v : w) if (v) return false;
        return true;
    }

    bool msb()  {
        return (w[WordCount - 1] >> 63) & 1;
    }

    // ===== Comparison =====
    int compare( RegisterN& other)  {
        for (int i = WordCount - 1; i >= 0; --i) {
            if (w[i] < other.w[i]) return -1;
            if (w[i] > other.w[i]) return  1;
        }
        return 0;
    }

    // ===== Arithmetic =====
    RegisterN operator+( RegisterN& rhs)  {
        RegisterN r;
        __uint128_t carry = 0;
        for (size_t i = 0; i < WordCount; ++i) {
            __uint128_t sum = (__uint128_t)w[i] + rhs.w[i] + carry;
            r.w[i] = (uint64_t)sum;
            carry  = sum >> 64;
        }
        return r;
    }

    RegisterN operator-( RegisterN& rhs)  {
        RegisterN r;
        __uint128_t borrow = 0;
        for (size_t i = 0; i < WordCount; ++i) {
            __uint128_t lhs = (__uint128_t)w[i];
            __uint128_t rhsb = (__uint128_t)rhs.w[i] + borrow;
            if (lhs >= rhsb) {
                r.w[i] = (uint64_t)(lhs - rhsb);
                borrow = 0;
            } else {
                r.w[i] = (uint64_t)(((__uint128_t(1) << 64) + lhs) - rhsb);
                borrow = 1;
            }
        }
        return r;
    }

    // ===== Bitwise =====
    RegisterN operator&( RegisterN& rhs)  {
        RegisterN r;
        for (size_t i = 0; i < WordCount; ++i) r.w[i] = w[i] & rhs.w[i];
        return r;
    }

    RegisterN operator|( RegisterN& rhs)  {
        RegisterN r;
        for (size_t i = 0; i < WordCount; ++i) r.w[i] = w[i] | rhs.w[i];
        return r;
    }

    RegisterN operator^( RegisterN& rhs)  {
        RegisterN r;
        for (size_t i = 0; i < WordCount; ++i) r.w[i] = w[i] ^ rhs.w[i];
        return r;
    }

    RegisterN operator~()  {
        RegisterN r;
        for (size_t i = 0; i < WordCount; ++i) r.w[i] = ~w[i];
        return r;
    }

    // ===== Shifts =====
    RegisterN shl(size_t bits)  {
        RegisterN r;
        size_t wordShift = bits / 64;
        size_t bitShift  = bits % 64;

        for (int i = WordCount - 1; i >= 0; --i) {
            uint64_t v = 0;
            int src = i - (int)wordShift;
            if (src >= 0) {
                v = w[src] << bitShift;
                if (bitShift && src > 0)
                    v |= w[src - 1] >> (64 - bitShift);
            }
            r.w[i] = v;
        }
        return r;
    }

    RegisterN shr(size_t bits)  {
        RegisterN r;
        size_t wordShift = bits / 64;
        size_t bitShift  = bits % 64;

        for (size_t i = 0; i < WordCount; ++i) {
            uint64_t v = 0;
            int src = i + wordShift;
            if (src < (int)WordCount) {
                v = w[src] >> bitShift;
                if (bitShift && src + 1 < (int)WordCount)
                    v |= w[src + 1] << (64 - bitShift);
            }
            r.w[i] = v;
        }
        return r;
    }

    // ===== Multiply (low Bits only) =====
    RegisterN mul( RegisterN& rhs)  {
        RegisterN r;
        std::array<__uint128_t, WordCount * 2> tmp{};
        tmp.fill(0);

        for (size_t i = 0; i < WordCount; ++i)
            for (size_t j = 0; j < WordCount; ++j)
                tmp[i + j] += (__uint128_t)w[i] * rhs.w[j];

        __uint128_t carry = 0;
        for (size_t i = 0; i < WordCount; ++i) {
            __uint128_t val = tmp[i] + carry;
            r.w[i] = (uint64_t)val;
            carry = val >> 64;
        }
        return r;
    }

    // ===== Hex I/O =====
    std::string toHex()  {
        std::ostringstream oss;
        oss << "0x";
        bool started = false;
        for (int i = WordCount - 1; i >= 0; --i) {
            if (!started) {
                if (w[i] == 0) continue;
                oss << std::hex << std::uppercase << w[i];
                started = true;
            } else {
                oss << std::setw(16) << std::setfill('0')
                    << std::hex << std::uppercase << w[i];
            }
        }
        if (!started) oss << "0";
        return oss.str();
    }

    static RegisterN fromHex(std::string s) {
        RegisterN r;
        if (s.rfind("0x", 0) == 0) s = s.substr(2);
        while (s.size() < WordCount * 16) s = "0" + s;
        for (size_t i = 0; i < WordCount; ++i) {
            std::string part = s.substr(s.size() - (i + 1) * 16, 16);
            r.w[i] = std::stoull(part, nullptr, 16);
        }
        return r;
    }

    // ===== Access =====
    uint64_t& operator[](size_t i)       { return w[i]; }
    uint64_t  operator[](size_t i) const { return w[i]; }
};
