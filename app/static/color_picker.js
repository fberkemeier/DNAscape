export default {
  template: `
    <div class="dn-color-picker" :data-dn-view="view">
      <q-color :key="view" v-model="color" no-footer
        :format-model="formatModel"
        :default-view="view === 'spectrum' ? 'spectrum' : 'palette'"
        :palette="view === 'dnascape' ? brandPalette : undefined"
        @change="commit" />
      <q-tabs v-model="view" dense align="justify" class="dn-color-picker-tabs">
        <q-tab name="palette" icon="style" :ripple="false" aria-label="Colour palette" />
        <q-tab name="spectrum" icon="gradient" :ripple="false" aria-label="Colour spectrum" />
        <q-tab name="dnascape" :ripple="false" aria-label="DNAscape colours">
          <img :src="brandIcon" alt="" class="dn-color-picker-logo" />
        </q-tab>
      </q-tabs>
    </div>`,
  props: {
    modelValue: String,
    formatModel: {type: String, default: 'hex'},
    defaultView: {type: String, default: 'palette'},
    brandPalette: Array,
    brandIcon: String,
  },
  emits: ['change'],
  data() {
    return {view: this.defaultView, color: this.modelValue};
  },
  watch: {
    modelValue(value) { this.color = value; },
  },
  methods: {
    commit(value) {
      this.color = value;
      this.$emit('change', value);
    },
  },
};
