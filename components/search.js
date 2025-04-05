Vue.component('search-component', {
  data: function() {
    return {
      searchTerm: '',
      searchResults: []
    }
  },
  props: ['json_data'],
  methods: {
    performSearch: function() {
      this.searchResults = searchAllAnnotations(this.json_data, this.searchTerm);
    },
    jumpToResult: function(result) {
      this.$emit('jump-to-time', result.time);
    }
  },
  template: `
    <div class="search-container">
      <div class="search-input-container">
        <input v-model="searchTerm" @keyup.enter="performSearch" placeholder="Search for objects, labels, or words...">
        <button @click="performSearch" class="search-button">Search</button>
      </div>
      <div v-if="searchResults.length > 0" class="search-results">
        <div v-for="result in searchResults" @click="jumpToResult(result)" class="search-result">
          <span>{{ result.type }}: {{ result.name }}</span>
          <span>{{ result.time.toFixed(2) }}s</span>
        </div>
      </div>
    </div>
  `
});
