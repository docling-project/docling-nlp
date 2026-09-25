//-*-C++-*-

#ifndef ANDROMEDA_UTILS_HASH_UTILS_H_
#define ANDROMEDA_UTILS_HASH_UTILS_H_

namespace andromeda
{
  namespace utils
  {

    // https://stackoverflow.com/questions/5085915/what-is-the-best-hash-function-for-uint64-t-keys-ranging-from-0-to-its-max-value
    static inline uint64_t murmerhash3(uint64_t val)
    {
      uint64_t hash = val;

      hash ^= hash >> 33;
      hash *= 0xff51afd7ed558ccd;
      hash ^= hash >> 33;
      hash *= 0xc4ceb9fe1a85ec53;
      hash ^= hash >> 33;

      return hash;
    }

    static inline uint64_t combine_hash(uint64_t hash, uint64_t val)
    {
      hash ^= val + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      return hash;
    }

    /*
    static uint64_t to_hash(const std::string& text)
    {
      const static std::string seed = "QWERTYUIOP!@#$%^&*()_+qwertyuiop";
      std::string name = seed + text;
      
      uint64_t hash = std::hash<std::string_view>{}(name.c_str());
      return hash;
    }
    */
    
    static uint64_t to_reproducible_hash(const std::string& text)
    {
      auto beg = text.begin();
      auto end = utf8::find_invalid(text.begin(), text.end());
      
      //std::vector<unsigned short> utf16line={};
      std::vector<uint16_t> utf16line={};
      utf8::utf8to16(beg, end, std::back_inserter(utf16line));

      uint64_t hash = utf16line.size();
      hash = utils::murmerhash3(hash);

      for(auto& utf16:utf16line)
	{
	  hash = utils::combine_hash(hash, utf16);
	}
      
      return hash;      
    }
    
    [[maybe_unused]] static uint64_t to_hash(const std::vector<uint64_t>& hashes)
    {
      switch(hashes.size())
        {
        case 0:
	  {
	    LOG_S(FATAL) << "hashing path of length 0";
	    return -1;
	  }
	  
        case 1:
	  {
	    return hashes.at(0);
	  }

        default:
          {
            uint64_t hash = hashes.size();
            hash = utils::murmerhash3(hash);

            for(uint64_t chash:hashes)
              {
                hash = utils::combine_hash(hash, chash);
              }

            return hash;
          }
        }
    }

    [[maybe_unused]] static uint16_t to_flvr_hash(const std::string& text)
    {
      const static std::string seed = "QWERTYUIOP!@#$%^&*()_+qwertyuiop";
      std::string name = seed + text;
      
      uint64_t uhash64 = std::hash<std::string_view>{}(name.c_str());
      uint16_t uhash16 = uhash64;
	
      return uhash16;
    }
    
  }

}

#endif
