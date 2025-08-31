# Quick Workflow

1. **Upload**
   * Upload an Excel file (`.xlsx` or `.xls`).
2. **(Optional) Configure date columns**
   * Inspect sample values from a chosen column.
   * Test parsing:
     * `auto`
     * `Excel serial`
     * `single format`
     * `multiple formats`
   * Apply parsing → column is treated as dates for later filtering.
3. **Select columns for analysis**
   * Choose one or more columns as filter dimensions.
4. **Configure filters / thresholds**
   * For each selected column, pick a filter type:
     * `date`
     * `numeric`
     * `categorical`
   * Configure parameters (see filter types below).
5. **Choose output settings**
   * Pick an **ID column** (used to list matching records).
   * Select numeric columns for statistics (or "all numeric").
6. **Set analysis controls**
   * Configure **Minimum Matching Rows** (ignore tiny/noisy groups).
7. **Run analysis**
   * The app enumerates combinations of selected columns.
   * Applies all configured filter variations.
   * Conditions combine with  **AND logic** .
   * Results → table with conditions, counts, IDs, and statistics.
8. **Download**
   * Export results to Excel (`.xlsx`).
   * Includes results table + summary rows (counts, sums, variance).

---

# Filter Types

### Date filters

* **Range** : between Start & End (inclusive).
* **Before** : earlier than date.
* **After** : later than date.
* **On** : exactly that date.
* **Last N Days** : from `(latest − N)` to latest.
* **Next N Days** : from earliest up to `(earliest + N)`.
* **Date parsing options** : `auto`, `Excel serial`, `single format`, `multiple formats`.

---

### Numeric filters

* **Mean / Median** : split at column mean or median.
* **Custom** : supply threshold manually.
* **Greater Than** : `> threshold` (can supply multiple).
* **Less Than** : `< threshold` (can supply multiple).
* **Range Divisions** : split numeric span into N ranges.
* **Multiple thresholds** : test across several cut points.

---

### Categorical filters

* **Select Specific Values** : include chosen values (individually + grouped).
* **Select All Values** : test each unique value + all together.
* **Select Top N Values** : limit to N most frequent (individually + grouped).

---



# How Combinations / Logic Work